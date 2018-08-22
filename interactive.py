import argparse
import code
import json
import logging
import os
import prettytable
import time
import torch
import torch.nn as nn

from torch.serialization import default_restore_location

from tqdm import tqdm
from reader import models, utils
from reader.data.dictionary import Dictionary
from reader.data.dataset import ReadingDataset, BatchSampler
from reader.data.tokenizer import Tokenizer, SpacyTokenizer, CoreNLPTokenizer


def get_args():
    parser = argparse.ArgumentParser('Question Answering - Interactive Console')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--tokenizer', default='corenlp', choices=['spacy', 'corenlp'], help='tokenizer')
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)

    # Load arguments from checkpoint (no need to load pretrained embeddings)
    state_dict = torch.load(args.checkpoint, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args = argparse.Namespace(**{**vars(state_dict['args']), **vars(args), 'embed_path': None})
    utils.init_logging(args)

    # Load dictionary and pretrained model
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    logging.info('Loaded a dictionary with {} words'.format(len(dictionary)))
    model = models.build_model(args, dictionary).cuda().eval()
    model.load_state_dict(state_dict['model'])
    with open(os.path.join(args.data, 'feature_dict.json')) as file:
        feature_dict = json.load(file)

    if args.tokenizer == 'spacy':
        tokenizer = SpacyTokenizer(annotators=['lemma', 'pos', 'ner'])
    elif args.tokenizer == 'corenlp':
        tokenizer = CoreNLPTokenizer(annotators=['lemma', 'pos', 'ner'])

    def answer(context, question, topk=1):
        t0 = time.time()
        # Tokenize context and question
        context = tokenizer.tokenize(context)
        question = tokenizer.tokenize(question)
        examples = [{'id': 0, 'question': question, 'context_id': 0, 'answers': {'spans': [], 'texts': []}}]

        test_dataset = ReadingDataset([context], examples, dictionary, feature_dict=feature_dict, skip_no_answer=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, num_workers=args.num_workers, collate_fn=test_dataset.collater,
            batch_sampler=BatchSampler(test_dataset, args.max_tokens, args.batch_size, shuffle=False, seed=args.seed)
        )

        # Forward pass
        with torch.no_grad():
            sample = utils.move_to_cuda(next(iter(test_loader)))
            start_scores, end_scores = model(sample['context_tokens'], sample['question_tokens'], context_features=sample['context_features'])
            # start_scores, end_scores = model(context_tokens.unsqueeze(0), question_tokens.unsqueeze(0))
            start_preds, end_preds, scores = model.decode(start_scores, end_scores, topk=topk)

        # Map predictions to span
        table = prettytable.PrettyTable(['Rank', 'Span', 'Score'])
        for i, (start_pred, end_pred, score) in enumerate(zip(start_preds[0], end_preds[0], scores[0])):
            start_idx = context['offsets'][start_pred][0]
            end_idx = context['offsets'][end_pred][1]
            text_pred = context['text'][start_idx: end_idx]
            table.add_row([i + 1, text_pred, score.item()])

        print(table)
        print('Time: %.4f' % (time.time() - t0))

    # Read-eval-print loop
    code.interact(banner='>>> Usage: answer(context, question, topk=5)', local=locals())


if __name__ == '__main__':
    args = get_args()
    main(args)

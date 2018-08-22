import argparse
import json
import logging
import os
import torch
import torch.nn as nn

from torch.serialization import default_restore_location

from tqdm import tqdm
from reader import models, utils
from reader.data.dictionary import Dictionary
from reader.data.dataset import ReadingDataset, BatchSampler


def get_args():
    parser = argparse.ArgumentParser('Question Answering - Prediction')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')
    parser.add_argument('--input', required=True, help='processed JSON file to be evaluated')
    parser.add_argument('--output', required=True, help='output file containing all predictions')
    parser.add_argument('--feature-dict', required=True, help='feature dictionary for context words')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--max-tokens', default=32000, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--num-workers', default=4, type=int, help='number of data workers')
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

    # Load dataset
    with open(args.input) as input_file, open(args.feature_dict) as feature_file:
        contents, feature_dict = json.load(input_file), json.load(feature_file)

    test_dataset = ReadingDataset(
        contents['contexts'], contents['examples'], dictionary, feature_dict=feature_dict,
        skip_no_answer=False, single_answer=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=test_dataset.collater,
        batch_sampler=BatchSampler(test_dataset, args.max_tokens, args.batch_size, shuffle=False, seed=args.seed)
    )

    progress_bar = tqdm(test_loader, desc='| Prediction', leave=False)
    stats = {'num_tokens': 0., 'batch_size': 0.}
    results = {}

    for batch_id, sample in enumerate(progress_bar):
        sample = utils.move_to_cuda(sample) if not args.no_cuda else sample
        with torch.no_grad():
            start_scores, end_scores = model(sample['context_tokens'], sample['question_tokens'], context_features=sample['context_features'])
            start_preds, end_preds, _ = model.decode(start_scores, end_scores, max_len=15)
            stats['num_tokens'] += sample['num_tokens']
            stats['batch_size'] += len(sample['id'])

            for i, (start_pred, end_pred) in enumerate(zip(start_preds, end_preds)):
                context = test_dataset.contexts[test_dataset.context_ids[sample['id'][i]]]
                start_idx = context['offsets'][start_pred][0]
                end_idx = context['offsets'][end_pred][1]
                text_pred = context['text'][start_idx: end_idx]
                results[sample['id'][i]] = text_pred

        progress_bar.set_postfix({
            'num_tokens': '{:.3g}'.format(stats['num_tokens'] / stats['batch_size']),
            'batch_size': '{:.3g}'.format(stats['batch_size'] / (batch_id + 1)),
        }, refresh=True)

    with open(args.output, 'w') as file:
        json.dump(results, file)
        logging.info('Writing predictions to {}'.format(args.output))


if __name__ == '__main__':
    args = get_args()
    main(args)

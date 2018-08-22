import argparse
import itertools
import json
import logging
import os

from multiprocessing import Pool
from reader import utils
from reader.data.tokenizer import Tokenizer, SpacyTokenizer, CoreNLPTokenizer
from reader.data.dictionary import Dictionary


# Global tokenizer
TOK = None


# Each process has its own tokenizer
def init_tokenizer(tokenizer, annotators):
    global TOK
    if tokenizer == 'spacy':
        TOK = SpacyTokenizer(annotators=annotators)
    elif tokenizer == 'corenlp':
        TOK = CoreNLPTokenizer(annotators=annotators)


# Multiprocessing requires global function
def tokenize(text):
    global TOK
    return TOK.tokenize(text)


def tokenize_all(texts, tokenizer, annotators, num_workers=None):
    """Tokenization might take a long time, even when done in parallel"""
    workers = Pool(num_workers, init_tokenizer, initargs=[tokenizer, annotators])
    tokens = workers.map(tokenize, texts)
    workers.close()
    workers.join()
    return tokens


def get_args():
    parser = argparse.ArgumentParser('Pre-processing SQuAD datasets')
    parser.add_argument('--data', required=True, help='dataset path')
    parser.add_argument('--dest-dir', default='data', help='destination dir')
    parser.add_argument('--tokenizer', default='corenlp', choices=['spacy', 'corenlp'], help='tokenizer')
    parser.add_argument('--num-workers', type=int, default=None, help='number of workers for tokenization')

    parser.add_argument('--threshold', default=0, type=int, help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--embed-path', type=str, default=None, help='path to pretrained embedding')
    parser.add_argument('--restrict-vocab', action='store_true', help='only use pre-trained words in embedding_file')
    return parser.parse_args()


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    for split in ['train', 'dev']:
        # Load JSON dataset
        dataset_path = os.path.join(args.data, '{}-v1.1.json'.format(split))
        dataset = load_dataset(dataset_path)
        logging.info('Loaded dataset {} ({} questions, {} contexts)'.format(
            dataset_path, len(dataset['questions']), len(dataset['contexts'])))

        # Tokenize questions and contexts and build a dictionary
        questions = tokenize_all(dataset['questions'], args.tokenizer, ['lemma'], args.num_workers)
        contexts = tokenize_all(dataset['contexts'], args.tokenizer, ['lemma', 'pos', 'ner'], args.num_workers)

        # Build a dictionary from train examples only
        if split == 'train':
            build_dictionary(args, [q['tokens'] for q in questions], [c['tokens'] for c in contexts])

        examples = []
        for qid, cid in enumerate(dataset['context_ids']):
            answer_spans, answer_texts = [], []
            for ans in dataset['answers'][qid]:
                # Map answers to token spans
                char_start, char_end = ans['answer_start'], ans['answer_start'] + len(ans['text'])
                token_start = [i for i, tok in enumerate(contexts[cid]['offsets']) if tok[0] == char_start]
                token_end = [i for i, tok in enumerate(contexts[cid]['offsets']) if tok[1] == char_end]

                # Bad tokenization can lead to no answer found
                if len(token_start) == 1 and len(token_end) == 1:
                    answer_spans.append((token_start[0], token_end[0]))
                    answer_texts.append(ans['text'])

            examples.append({
                'id': dataset['question_ids'][qid],
                'answers': {'spans': answer_spans, 'texts': answer_texts},
                'question': {key: questions[qid][key] for key in ['tokens', 'lemma']},
                'context_id': cid,
            })

        # Write preprocessed data to file
        output = {'contexts': contexts, 'examples': examples}
        output_file = os.path.join(args.dest_dir, '%s.json' % split)
        with open(output_file, 'w') as file:
            json.dump(output, file)
            logging.info('Wrote {} examples to {}'.format(len(examples), output_file))


def load_dataset(filename):
    """Parse a JSON file into a dictionary"""
    with open(filename, 'r') as file:
        data = json.load(file)['data']
    outputs = {'question_ids': [], 'questions': [], 'answers': [], 'contexts': [], 'context_ids': []}
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                outputs['question_ids'].append(qa['id'])
                outputs['questions'].append(qa['question'])
                outputs['answers'].append(qa['answers'])
                outputs['context_ids'].append(len(outputs['contexts']))
            outputs['contexts'].append(paragraph['context'])
    return outputs


def build_dictionary(args, question_tokens, context_tokens):
    """Build a dictionary from questions and contexts"""
    valid_words = None
    if args.restrict_vocab and args.embed_path is not None:
        with open(args.embed_path) as file:
            valid_words = {Tokenizer.normalize(line.rstrip().split(' ')[0]) for line in file}

    dictionary = Dictionary()
    for text in itertools.chain(question_tokens, context_tokens):
        for word in text:
            if valid_words is None or word in valid_words:
                dictionary.add_word(word)

    dictionary.finalize(threshold=args.threshold, num_words=args.num_words)
    dictionary.save(os.path.join(args.dest_dir, 'dict.txt'))
    logging.info('Built a dictionary with {} words'.format(len(dictionary)))
    return dictionary


if __name__ == '__main__':
    args = get_args()
    utils.init_logging(args)
    main(args)

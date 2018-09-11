import collections
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class ReadingDataset(Dataset):
    def __init__(
        self, args, contexts, examples, dictionary, char_dictionary,
        feature_dict=None, single_answer=False, skip_no_answer=False,
    ):
        self.args = args
        self.contexts = contexts
        self.dictionary = dictionary
        self.char_dictionary = char_dictionary
        self.single_answer = single_answer

        self.examples = [ex for ex in examples if len(ex['answers']['spans']) > 0] if skip_no_answer else examples
        self.feature_dict = self._build_feature_dict() if feature_dict is None else feature_dict

        # Save context ids to avoid storing contexts inside examples
        self.context_ids = {ex['id']: ex['context_id'] for ex in self.examples}

        # Save answer texts for official evaluations
        self.answer_texts = {ex['id']: ex['answers']['texts'] for ex in self.examples}

        # Calculate sizes for the sampler
        self.question_sizes = np.array([len(ex['question']['tokens']) for ex in self.examples])
        self.context_sizes = np.array([len(self.contexts[ex['context_id']]['tokens']) for ex in self.examples])

    def _build_feature_dict(self):
        """Build a dictionary that maps feature names to indices"""
        feature_dict = collections.OrderedDict()
        def maybe_insert(attribute, features):
            if eval(getattr(self.args, attribute, 'False')):
                for feature in features:
                    if feature not in feature_dict:
                        feature_dict[feature] = len(feature_dict)

        maybe_insert('use_in_question', ['in_question_cased', 'in_question_uncased'])
        maybe_insert('use_lemma', ['in_question_lemma'])
        maybe_insert('use_pos', ['pos={}'.format(pos) for ctx in self.contexts for pos in ctx['pos']])
        maybe_insert('use_ner', ['ner={}'.format(ner) for ctx in self.contexts for ner in ctx['ner']])
        maybe_insert('use_tf', ['tf'])

        # Initializing RNNs requires known input size
        self.args.num_features = len(feature_dict)

        # Save feature dictionary for later use with evaluation
        with open(os.path.join(self.args.data, 'feature_dict.json'), 'w') as file:
            json.dump(feature_dict, file)
        return feature_dict

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        context = self.contexts[example['context_id']]
        question_tokens = torch.LongTensor([self.dictionary.index(w) for w in example['question']['tokens']])
        context_tokens = torch.LongTensor([self.dictionary.index(w) for w in context['tokens']])

        question_chars = [torch.LongTensor([self.char_dictionary.index(c) for c in w]) for w in example['question']['tokens']]
        context_chars = [torch.LongTensor([self.char_dictionary.index(c) for c in w]) for w in context['tokens']]

        if len(self.feature_dict) == 0:
            context_features = None
        else:
            feature_dict = self.feature_dict
            context_features = torch.zeros(self.context_sizes[index], len(feature_dict))
            cased_question = set(example['question']['tokens'])
            uncased_question = set(w.lower() for w in example['question']['tokens'])
            lemma_question = set(example['question']['lemma'])
            counter = collections.Counter([w.lower() for w in context['tokens']])

            # Add features for each context word
            for i, context_word in enumerate(context['tokens']):
                if 'in_question_cased' in feature_dict and context_word in cased_question:
                    context_features[i][feature_dict['in_question_cased']] = 1.0
                if 'in_question_uncased' in feature_dict and context_word in uncased_question:
                    context_features[i][feature_dict['in_question_uncased']] = 1.0
                if 'in_question_lemma' in feature_dict and context['lemma'][i] in lemma_question:
                    context_features[i][feature_dict['in_question_lemma']] = 1.0
                if 'pos={}'.format(context['pos'][i]) in feature_dict:
                    context_features[i][feature_dict['pos={}'.format(context['pos'][i])]] = 1.0
                if 'ner={}'.format(context['ner'][i]) in feature_dict:
                    context_features[i][feature_dict['ner={}'.format(context['ner'][i])]] = 1.0
                if 'tf' in feature_dict:
                    context_features[i][feature_dict['tf']] = counter[context_word.lower()] / self.context_sizes[index]

        if len(example['answers']['spans']) == 0:
            answer_start, answer_end = None, None
        else:
            num_answers = 1 if self.single_answer else len(example['answers']['spans'])
            answer_start = [ans[0] for ans in example['answers']['spans'][:num_answers]]
            answer_end = [ans[1] for ans in example['answers']['spans'][:num_answers]]

        return {
            'id': example['id'],
            'question_tokens': question_tokens,
            'context_tokens': context_tokens,
            'question_chars': question_chars,
            'context_chars': context_chars,
            'context_features': context_features,
            'answer_start': answer_start,
            'answer_end': answer_end,
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch"""
        if len(samples) == 0:
            return {}

        def merge(values):
            # Convert a list of tensors into a padded tensor
            max_length = max(v.size(0) for v in values)
            result = values[0].new(len(values), max_length, *values[0].size()[1:])
            result = result.fill_(self.dictionary.pad_idx)
            for i, v in enumerate(values):
                result[i, :len(v)].copy_(v)
            return result

        def aggregate(list_values):
            # Convert a list of lists of tensors into a padded tensor
            seq_length = max_length = 0
            for values in list_values:
                seq_length = max(seq_length, len(values))
                max_length = max(max_length, max(v.size(0) for v in values))
            result = list_values[0][0].new(len(list_values), seq_length, max_length)
            result = result.fill_(self.dictionary.pad_idx)
            for i, values in enumerate(list_values):
                for j, v in enumerate(values):
                    result[i, j, :len(v)].copy_(v)
            return result

        id = [s['id'] for s in samples]
        question_tokens = merge([s['question_tokens'] for s in samples])
        context_tokens = merge([s['context_tokens'] for s in samples])
        question_chars = aggregate([s['question_chars'] for s in samples])
        context_chars = aggregate([s['context_chars'] for s in samples])
        answer_start = [s['answer_start'] for s in samples]
        answer_end = [s['answer_end'] for s in samples]

        # Sort by descending context lengths
        context_lengths = torch.LongTensor([s['context_tokens'].numel() for s in samples])
        context_lengths, sort_order = context_lengths.sort(descending=True)
        question_tokens = question_tokens.index_select(0, sort_order)
        context_tokens = context_tokens.index_select(0, sort_order)
        question_chars = question_chars.index_select(0, sort_order)
        context_chars = context_chars.index_select(0, sort_order)

        if samples[0]['context_features'] is None:
            context_features = None
        else:
            context_features = merge([s['context_features'] for s in samples])
            context_features = context_features.index_select(0, sort_order)

        id = [id[i] for i in sort_order]
        answer_start = [answer_start[i] for i in sort_order]
        answer_end = [answer_end[i] for i in sort_order]

        return {
            'id': id,
            'question_tokens': question_tokens,
            'context_tokens': context_tokens,
            'question_chars': question_chars,
            'context_chars': context_chars,
            'context_features': context_features,
            'answer_start': answer_start,
            'answer_end': answer_end,
            'num_tokens': sum(len(s['context_tokens']) for s in samples),
        }


class BatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=None, batch_size=None, shuffle=True, seed=42):
        self.dataset, self.shuffle, self.seed = dataset, shuffle, seed
        self.batch_size = batch_size if batch_size is not None else float('Inf')
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.batches = self._batch_generator()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _batch_generator(self):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        indices = indices[np.argsort(self.dataset.question_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.dataset.context_sizes[indices], kind='mergesort')]

        batches, batch, sample_len = [], [], 0
        for idx in indices:
            batch.append(idx)
            sample_len = max(sample_len, self.dataset.context_sizes[idx])
            num_tokens = len(batch) * sample_len
            if len(batch) == self.batch_size or num_tokens > self.max_tokens:
                batches.append(batch)
                batch, sample_len = [], 0
        if len(batch) > 0:
            batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)
        return batches

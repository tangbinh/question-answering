import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F

from tqdm import tqdm
from reader import models, utils
from reader.data.dictionary import Dictionary
from reader.data.dataset import ReadingDataset, BatchSampler
from reader.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    parser = argparse.ArgumentParser('Question Answering - Training')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='data', help='path to data directory')
    parser.add_argument('--max-tokens', default=16000, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--num-workers', default=4, type=int, help='number of data workers')

    # Add model arguments
    parser.add_argument('--arch', default='drqa', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=50, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=10, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.99, type=float, help='momentum factor')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--lr-shrink', default=0.1, type=float, help='learning rate shrink factor for annealing')
    parser.add_argument('--min-lr', default=1e-6, type=float, help='minimum learning rate')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported.')
    torch.manual_seed(args.seed)
    utils.init_logging(args)

    # Load a dictionary
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    logging.info('Loaded a word dictionary with {} words'.format(len(dictionary)))
    char_dictionary = Dictionary.load(os.path.join(args.data, 'char_dict.txt'))
    logging.info('Loaded a character dictionary with {} words'.format(len(char_dictionary)))

    # Load a training and validation dataset
    with open(os.path.join(args.data, 'train.json')) as file:
        train_contents = json.load(file)
        train_dataset = ReadingDataset(
            args, train_contents['contexts'], train_contents['examples'], dictionary,
            char_dictionary, skip_no_answer=True, single_answer=True,
        )
        logging.info('Created a training dataset of {} examples'.format(len(train_dataset)))

    with open(os.path.join(args.data, 'dev.json')) as file:
        contents = json.load(file)
        valid_dataset = ReadingDataset(
            args, contents['contexts'], contents['examples'], dictionary, char_dictionary,
            feature_dict=train_dataset.feature_dict, skip_no_answer=True, single_answer=True
        )
        logging.info('Created a validation dataset of {} examples'.format(len(valid_dataset)))

    train_dataset.collater([train_dataset[2], train_dataset[3], train_dataset[7]])

    # Build a model
    model = models.build_model(args, dictionary, char_dictionary).cuda()
    logging.info('Built a model with {} parameters'.format(sum(p.numel() for p in model.parameters())))

    # Build an optimizer and a learning rate schedule
    optimizer = torch.optim.Adamax(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=args.lr_shrink)

    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer, lr_scheduler)
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1

    for epoch in range(last_epoch + 1, args.max_epoch):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=args.num_workers, collate_fn=train_dataset.collater,
            batch_sampler=BatchSampler(train_dataset, args.max_tokens, args.batch_size, shuffle=True, seed=args.seed)
        )

        model.train()
        stats = {'loss': 0., 'lr': 0., 'num_tokens': 0., 'batch_size': 0., 'grad_norm': 0., 'clip': 0.}
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

        for batch_id, sample in enumerate(progress_bar):
            # Forward and backward pass
            sample = utils.move_to_cuda(sample)
            start_scores, end_scores = model(
                sample['context_tokens'], sample['question_tokens'],
                context_chars=sample['context_chars'],
                question_chars=sample['question_chars'],
                context_features=sample['context_features']
            )
            start_loss = F.nll_loss(start_scores, torch.LongTensor(sample['answer_start']).view(-1).cuda())
            end_loss = F.nll_loss(end_scores, torch.LongTensor(sample['answer_end']).view(-1).cuda())
            loss = start_loss + end_loss
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients and fix embeddings of infrequent words
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            if args.tune_embed is not None and hasattr(model, 'embedding'):
                model.embedding.weight.grad[args.tune_embed:] = 0
            optimizer.step()

            # Update statistics for progress bar
            stats['loss'] += loss.item()
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += sample['num_tokens'] / len(sample['id'])
            stats['batch_size'] += len(sample['id'])
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (batch_id + 1)) for key, value in stats.items()}, refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(value / len(progress_bar)) for key, value in stats.items())))

        # Adjust learning rate based on validation result
        f1_score = validate(args, model, valid_dataset, epoch)
        lr_scheduler.step(f1_score)

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, lr_scheduler, epoch, f1_score)
        if optimizer.param_groups[0]['lr'] <= args.min_lr:
            logging.info('Done training!')
            break


def validate(args, model, valid_dataset, epoch):
    model.eval()
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, num_workers=args.num_workers, collate_fn=valid_dataset.collater,
        batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, shuffle=True, seed=args.seed)
    )

    stats = {'start_acc': 0., 'end_acc': 0., 'token_match': 0., 'f1': 0., 'exact_match': 0., 'num_tokens': 0., 'batch_size': 0.}
    progress_bar = tqdm(valid_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

    for batch_id, sample in enumerate(progress_bar):
        sample = utils.move_to_cuda(sample)
        with torch.no_grad():
            start_scores, end_scores = model(
                sample['context_tokens'], sample['question_tokens'],
                context_chars=sample['context_chars'],
                question_chars=sample['question_chars'],
                context_features=sample['context_features']
            )
            start_target, end_target = sample['answer_start'], sample['answer_end']

            stats['num_tokens'] += sample['num_tokens']
            stats['batch_size'] += len(sample['id'])

            start_pred, end_pred, _ = model.decode(start_scores, end_scores, max_len=15)
            stats['start_acc'] += sum(ex_pred in ex_target for ex_pred, ex_target in zip(start_pred, start_target))
            stats['end_acc'] += sum(ex_pred in ex_target for ex_pred, ex_target in zip(end_pred, end_target))

            for i, (start_ex, end_ex) in enumerate(zip(start_pred, end_pred)):
                # Check if the pair of predicted tokens in the targets
                stats['token_match'] += any((start_ex == s and end_ex == t) for s, t in zip(start_target[i], end_target[i]))

                # Official evaluation
                text_target = valid_dataset.answer_texts[sample['id'][i]]
                context = valid_dataset.contexts[valid_dataset.context_ids[sample['id'][i]]]
                start_idx = context['offsets'][start_ex][0]
                end_idx = context['offsets'][end_ex][1]
                text_pred = context['text'][start_idx: end_idx]
                stats['exact_match'] += utils.metric_max_over_ground_truths(utils.exact_match_score, text_pred, text_target)
                stats['f1'] += utils.metric_max_over_ground_truths(utils.f1_score, text_pred, text_target)

        progress_bar.set_postfix({key: '{:.3g}'.format(value / (stats['batch_size'] if key != 'batch_size' else (batch_id + 1)))
            for key, value in stats.items()}, refresh=True)

    logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(
        value / (stats['batch_size'] if key != 'batch_size' else len(progress_bar))) for key, value in stats.items())))

    return stats['f1'] / stats['batch_size']


if __name__ == '__main__':
    args = get_args()
    main(args)

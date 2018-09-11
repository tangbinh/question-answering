import torch
import torch.nn as nn
import torch.nn.functional as F

from reader import utils
from reader.models.layers import StackedRNN, BilinearAttention, SequenceAttention, SelfAttention
from reader.models import ReadingModel, register_model, register_model_architecture


@register_model('drqa')
class DrQA(ReadingModel):
    def __init__(
        self, dictionary, embed_dim=300, hidden_size=128, context_layers=3, question_layers=3, dropout=0.4,
        bidirectional=True, concat_layers=True, question_embed=True, pretrained_embed=None, num_features=0,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self.question_embed = question_embed
        self.num_features = num_features

        self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)
        if pretrained_embed is not None:
            utils.load_embedding(self.embedding.weight.data, pretrained_embed, dictionary)

        if question_embed:
            self.context_question_attention = SequenceAttention(embed_dim)

        self.context_rnn = StackedRNN(
            input_size=(1 + question_embed) * embed_dim + num_features, hidden_size=hidden_size,
            num_layers=context_layers, dropout=dropout, bidirectional=bidirectional, concat_layers=concat_layers
        )

        self.question_rnn = StackedRNN(
            input_size=embed_dim, hidden_size=hidden_size, num_layers=question_layers,
            dropout=dropout, bidirectional=bidirectional, concat_layers=concat_layers
        )

        context_size = question_size = (1 + bidirectional) * hidden_size
        if concat_layers:
            context_size = context_size * context_layers
            question_size = question_size * question_layers

        self.question_attention = SelfAttention(question_size)
        self.start_attention = BilinearAttention(question_size, context_size)
        self.end_attention = BilinearAttention(question_size, context_size)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--embed-dim', type=int, help='embedding dimension')
        parser.add_argument('--embed-path', help='path to pretrained embeddings')
        parser.add_argument('--hidden-size', type=int, help='hidden dimension')
        parser.add_argument('--context-layers', type=int, help='number of context layers')
        parser.add_argument('--question-layers', type=int, help='number of question layers')
        parser.add_argument('--dropout', type=float, help='dropout probability')
        parser.add_argument('--bidirectional', type=str, help='bidirectional RNNs for questions and contexts')
        parser.add_argument('--concat-layers', type=str, help='concatenate outputs of RNN layers')
        parser.add_argument('--question-embed', type=str, help='attend over question words when calculating context embeddings')

        parser.add_argument('--use-in-question', type=str, help='include in-question features for context words')
        parser.add_argument('--use-lemma', type=str, help='include in-question features for lemmas of context words')
        parser.add_argument('--use-pos', type=str, help='use part-of-speech features for context words')
        parser.add_argument('--use-ner', type=str, help='use name-entity-recognition features for context words')
        parser.add_argument('--use-tf', type=str, help='use term-frequency features for context words')
        parser.add_argument('--tune-embed', type=int, help='number of most frequent words whose embeddings are tuned')

    @classmethod
    def build_model(cls, args, dictionary, char_dictionary):
        base_architecture(args)
        return cls(
            dictionary, embed_dim=args.embed_dim, hidden_size=args.hidden_size, context_layers=args.context_layers,
            question_layers=args.question_layers, dropout=args.dropout, bidirectional=eval(args.bidirectional),
            question_embed=eval(args.question_embed), concat_layers=eval(args.concat_layers),
            pretrained_embed=args.embed_path, num_features=args.num_features,
        )

    def forward(self, context_tokens, question_tokens, **kwargs):
        # Ignore padding when calculating attentions
        context_mask = torch.eq(context_tokens, self.dictionary.pad_idx)
        question_mask = torch.eq(question_tokens, self.dictionary.pad_idx)

        # Embed context and question words
        context_embeddings = self.embedding(context_tokens)
        question_embeddings = self.embedding(question_tokens)
        context_embeddings = F.dropout(context_embeddings, p=self.dropout, training=self.training)
        question_embeddings = F.dropout(question_embeddings, p=self.dropout, training=self.training)

        if self.question_embed:
            context_hiddens = self.context_question_attention(context_embeddings, question_embeddings)
            context_embeddings = torch.cat([context_embeddings, context_hiddens], dim=2)

        # Combine with engineered features
        if self.num_features > 0 and 'context_features' in kwargs:
            context_embeddings = torch.cat([context_embeddings, kwargs['context_features']], dim=2)

        # Encode context words and question words with RNNs
        context_hiddens = self.context_rnn(context_embeddings, context_mask)
        question_hiddens = self.question_rnn(question_embeddings, question_mask)

        # Summarize hidden states of question words into a vector
        attention_scores = self.question_attention(question_hiddens, log_probs=False)
        question_hidden = (attention_scores.unsqueeze(dim=2) * question_hiddens).sum(dim=1)

        # Predict answers with attentions
        start_scores = self.start_attention(question_hidden, context_hiddens, context_mask, log_probs=self.training)
        end_scores = self.end_attention(question_hidden, context_hiddens, context_mask, log_probs=self.training)
        return start_scores, end_scores


@register_model_architecture('drqa', 'drqa')
def base_architecture(args):
    args.embed_dim = getattr(args, 'embed_dim', 300)
    args.embed_path = getattr(args, 'embed_path', None)
    args.hidden_size = getattr(args, 'hidden_size', 128)
    args.context_layers = getattr(args, 'context_layers', 3)
    args.question_layers = getattr(args, 'question_layers', 3)
    args.dropout = getattr(args, 'dropout', 0.4)
    args.bidirectional = getattr(args, 'bidirectional', 'True')
    args.concat_layers = getattr(args, 'concat_layers', 'True')
    args.question_embed = getattr(args, 'question_embed', 'True')

    args.use_in_question = getattr(args, 'use_in_question', 'True')
    args.use_lemma = getattr(args, 'use_lemma', 'True')
    args.use_pos = getattr(args, 'use_pos', 'True')
    args.use_ner = getattr(args, 'use_ner', 'True')
    args.use_tf = getattr(args, 'use_tf', 'True')
    args.tune_embed = getattr(args, 'tune_embed', 1000)

import torch
import torch.nn as nn
import torch.nn.functional as F

from reader import utils
from reader.models.layers import StackedRNN, SelfAttention
from reader.models import ReadingModel, register_model, register_model_architecture


@register_model('bidaf')
class BidirectionalAttentionFlow(ReadingModel):
    def __init__(
        self, dictionary, char_dictionary, embed_dim=300, char_embed_dim=16, kernel_size=5, hidden_size=128,
        context_layers=1, question_layers=1, start_layers=2, end_layers=1, dropout=0.2, bidirectional=True,
        pretrained_embed=None,
    ):
        super().__init__(dictionary)
        self.char_dictionary = char_dictionary
        self.dropout = dropout

        self.char_embedding = nn.Embedding(len(char_dictionary), char_embed_dim, char_dictionary.pad_idx)
        self.char_conv = nn.Conv2d(1, hidden_size, kernel_size=(kernel_size, char_embed_dim))
        self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)
        if pretrained_embed is not None:
            utils.load_embedding(self.embedding.weight.data, pretrained_embed, dictionary)

        self.context_rnn = StackedRNN(embed_dim + hidden_size, hidden_size, context_layers, dropout, bidirectional, concat_layers=False)
        context_size = (1 + bidirectional) * hidden_size
        self.similarity_linear = nn.Linear(3 * context_size, 1)

        self.start_rnn = StackedRNN(4 * context_size, hidden_size, start_layers, dropout, bidirectional, concat_layers=False)
        self.start_attention = SelfAttention(5 * context_size)
        self.end_rnn = StackedRNN(context_size, hidden_size, end_layers, dropout, bidirectional, concat_layers=False)
        self.end_attention = SelfAttention(5 * context_size)

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
        parser.add_argument('--tune-embed', type=int, help='number of most frequent words whose embeddings are tuned')

    @classmethod
    def build_model(cls, args, dictionary, char_dictionary):
        base_architecture(args)
        return cls(
            dictionary, char_dictionary, embed_dim=args.embed_dim, hidden_size=args.hidden_size, context_layers=args.context_layers,
            question_layers=args.question_layers, dropout=args.dropout, bidirectional=eval(args.bidirectional),
            pretrained_embed=args.embed_path,
        )

    def forward(self, context_tokens, question_tokens, **kwargs):
        # Ignore padding when calculating attentions
        context_mask = torch.eq(context_tokens, self.dictionary.pad_idx)
        question_mask = torch.eq(question_tokens, self.dictionary.pad_idx)

        # # Compute character embeddings
        context_chars = F.dropout(self.char_embedding(kwargs['context_chars']), p=self.dropout, training=self.training)
        question_chars = F.dropout(self.char_embedding(kwargs['question_chars']), p=self.dropout, training=self.training)
        (batch_size, context_len, _, char_embed_dim), question_len = context_chars.size(), question_chars.size(1)

        context_chars = self.char_conv(context_chars.view(batch_size * context_len, -1, char_embed_dim).unsqueeze(dim=1)).squeeze(dim=-1)
        context_chars = F.max_pool1d(context_chars, context_chars.size(2)).squeeze(dim=-1).view(batch_size, context_len, -1)

        question_chars = self.char_conv(question_chars.view(batch_size * question_len, -1, char_embed_dim).unsqueeze(dim=1)).squeeze(dim=-1)
        question_chars = F.max_pool1d(question_chars, question_chars.size(2)).squeeze(dim=-1).view(batch_size, question_len, -1)

        # Compute word embeddings
        context_words = F.dropout(self.embedding(context_tokens), p=self.dropout, training=self.training)
        question_words = F.dropout(self.embedding(question_tokens), p=self.dropout, training=self.training)

        # Combine character embeddings and word embeddings
        context_hiddens = self.context_rnn(torch.cat([context_chars, context_words], dim=-1), context_mask)
        question_hiddens = self.context_rnn(torch.cat([question_chars, question_words], dim=-1), question_mask)

        # Calculate similarity matrix
        mask_size = (batch_size, context_len, question_len)
        mesh_size = (batch_size, context_len, question_len, context_hiddens.size(-1))
        context_mesh = context_hiddens.unsqueeze(dim=2).expand(mesh_size)
        question_mesh = question_hiddens.unsqueeze(dim=1).expand(mesh_size)
        attention_scores = self.similarity_linear(torch.cat([context_mesh, question_mesh, context_mesh * question_mesh], dim=-1)).squeeze(dim=-1)
        attention_scores.masked_fill_(context_mask.unsqueeze(dim=2).expand(mask_size) & question_mask.unsqueeze(dim=1).expand(mask_size), float('-inf'))

        # Context-to-question attention and question-to-context attention
        question_weighted = F.softmax(attention_scores, dim=2).bmm(question_hiddens)
        context_weighted = F.softmax(attention_scores.max(dim=2)[0], dim=1).unsqueeze(dim=2) * context_hiddens
        context_hiddens = torch.cat([context_hiddens, question_weighted, context_hiddens * question_weighted, context_hiddens * context_weighted], dim=-1)
        context_hiddens = F.dropout(context_hiddens, p=self.dropout, training=self.training)

        # Predict answers
        start_hiddens = self.start_rnn(context_hiddens, context_mask)
        start_scores = self.start_attention(torch.cat([context_hiddens, start_hiddens], dim=-1), context_mask, log_probs=self.training)
        end_hiddens = self.end_rnn(start_hiddens, context_mask)
        end_scores = self.end_attention(torch.cat([context_hiddens, end_hiddens], dim=-1), context_mask, log_probs=self.training)
        return start_scores, end_scores


@register_model_architecture('bidaf', 'bidaf')
def base_architecture(args):
    args.embed_dim = getattr(args, 'embed_dim', 300)
    args.embed_path = getattr(args, 'embed_path', None)
    args.hidden_size = getattr(args, 'hidden_size', 128)
    args.context_layers = getattr(args, 'context_layers', 1)
    args.question_layers = getattr(args, 'question_layers', 1)
    args.dropout = getattr(args, 'dropout', 0.2)
    args.bidirectional = getattr(args, 'bidirectional', 'True')
    args.tune_embed = getattr(args, 'tune_embed', 1000)

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearAttention(nn.Module):
    def __init__(self, input_size, context_size):
        super().__init__()
        self.linear = nn.Linear(input_size, context_size)

    def forward(self, input, context, context_mask=None, log_probs=True):
        # input:    batch_size x input_size
        # context:  batch_size x context_length x context_size
        # output:   batch_size x context_length
        attn_scores = context.bmm(self.linear(input).unsqueeze(dim=2)).squeeze(dim=2)
        if context_mask is not None:
            attn_scores.masked_fill_(context_mask, float('-inf'))
        if log_probs:
            return F.log_softmax(attn_scores, dim=-1)
        else:
            return F.softmax(attn_scores, dim=-1)


class SequenceAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, context, context_mask=None):
        # input:    batch_size x input_size x hidden_size
        # context:  batch_size x context_size x hidden_size
        # output:   batch_size x input_size x hidden_size
        input_hiddens = F.relu(self.linear(input))
        context_hiddens = F.relu(self.linear(context))

        # Attend over context words when calculating hidden state for each input word
        input_scores = input_hiddens.bmm(context_hiddens.transpose(1, 2))
        if context_mask is not None:
            input_scores.masked_fill_(context_mask.unsqueeze(dim=1).expand(input_scores.size()), float('-inf'))
        input_scores = F.softmax(input_scores, dim=2)
        input_hiddens = input_scores.bmm(context_hiddens)
        return input_hiddens


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, input_mask=None):
        # input:    batch_size x input_size x hidden_size
        # output:   batch_size x hidden_size
        input_scores = self.linear(input).squeeze(dim=-1)
        if input_mask is not None:
            input_scores.masked_fill_(input_mask, float('-inf'))
        input_scores = F.softmax(input_scores, dim=-1)
        input_hidden = (input_scores.unsqueeze(dim=2) * input).sum(dim=1)
        return input_hidden


class StackedRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers=1, dropout=0, rnn_type=nn.LSTM,
        bidirectional=True, concat_layers=True
    ):
        super().__init__()
        self.dropout = dropout
        self.concat_layers = concat_layers

        self.layers = nn.ModuleList([
            rnn_type(input_size if layer == 0 else (1 + bidirectional) * hidden_size, hidden_size, bidirectional=bidirectional)
            for layer in range(num_layers)
        ])

    def forward(self, x, mask):
        # PackedPadded requires sequence to be sorted by lengths
        lengths = mask.eq(0).long().sum(dim=1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x = x.index_select(dim=0, index=idx_sort)
        lengths = lengths[idx_sort]

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        rnn_outputs = []
        for rnn in self.layers:
            if self.dropout > 0:
                dropout = F.dropout(rnn_input.data, p=self.dropout, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout, rnn_input.batch_sizes)
            rnn_input = rnn(rnn_input)[0]
            rnn_outputs.append(rnn_input)

        # Unpack output
        rnn_outputs = [nn.utils.rnn.pad_packed_sequence(o)[0] for o in rnn_outputs]
        output = torch.cat(rnn_outputs, dim=2) if self.concat_layers else rnn_outputs[-1]

        # T x B x C -> B x T x C
        output = output.transpose(0, 1)
        output = output.index_select(dim=0, index=idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = output.new(output.size(0), mask.size(1) - output.size(1), output.size(2)).zero_()
            output = torch.cat([output, padding], 1)

        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        return output

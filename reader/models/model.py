import logging
import torch
import torch.nn as nn

from reader.data.dictionary import Dictionary


class ReadingModel(nn.Module):
    def __init__(self, dictionary, embed_dim):
        super().__init__()
        self.dictionary = dictionary

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    def forward(self, context_tokens, question_tokens, **kwargs):
        """Return a start score and an end score for each token in context"""
        raise NotImplementedError


    @staticmethod
    def decode(start_scores, end_scores, topk=1, max_len=None):
        """Take argmax of constrained start_scores * end_scores."""
        pred_start, pred_end, pred_score = [], [], []
        max_len = max_len or start_scores.size(1)

        for i in range(start_scores.size(0)):
            # Outer product of scores to get a full matrix
            scores = torch.ger(start_scores[i], end_scores[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            max_scores, max_idx = scores.view(-1).topk(topk)
            pred_score.append(max_scores)
            pred_start.append(max_idx / scores.size(0))
            pred_end.append(max_idx % scores.size(0))
        return pred_start, pred_end, pred_score

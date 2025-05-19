
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class FeedForwardNNBuilder:

    def __init__(self, input_dim):
        self.layers = []
        self.prev_dim = input_dim
        self._built = False

    def _add_hidden_layer(
        self,
        hidden_dim,
        activation
    ):

        self._check_not_built()
        self.layers.append(nn.Linear(self.prev_dim, hidden_dim))
        self.layers.append(activation())
        self.prev_dim = hidden_dim
        return self

    def _add_output_layer(
        self,
        output_dim,
        activation
    ):

        if self._built:
            raise RuntimeError("Output layer has already been added.")
        self.layers.append(nn.Linear(self.prev_dim, output_dim))
        if activation:
            self.layers.append(activation())
        model = nn.Sequential(*self.layers)
        self._built = True
        return model

    def _add_dropout(self, p: float = 0.5):
        """
        Add dropout with probability p.
        """
        self._check_not_built()
        self.layers.append(nn.Dropout(p))
        return self

    def _add_batch_norm(self):
        """
        Add batch normalization on the last hidden dimension.
        """
        self._check_not_built()
        self.layers.append(nn.BatchNorm1d(self.prev_dim))
        return self

    def _check_not_built(self):
        if self._built:
            raise RuntimeError("Cannot modify builder after output layer is added.")


def invoke(input_dim):
    builder = FeedForwardNNBuilder(input_dim)
    return builder


meta = {
    "name": "FFN Input Layer",
    "min_args": 1,
    "max_args": 1
}

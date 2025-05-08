import torch
import torch.nn as nn
from torchinfo import summary


class LastTimeStep(nn.Module):

    def __init__(self, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first

    def forward(self, x):

        if self.batch_first:
            return x[:, -1, :]
        else:
            return x[-1, :, :]


class _RNNOutput(nn.Module):

    def forward(self, x):
        return x[0] if isinstance(x, tuple) else x


class RecurrentNNBuilder:
    """
    Builder for sequential RNN-based architectures.
    Supports SimpleRNN, LSTM, GRU layers, dropout, and final dense outputs.
    """
    def __init__(self, input_dim: int, batch_first: bool = True, seq_to_seq: bool = False):
        self.layers: list[nn.Module] = []
        self.prev_dim = input_dim
        self.batch_first = batch_first
        self.seq_to_seq = seq_to_seq
        self._built = False

    def _check_not_built(self):
        if self._built:
            raise RuntimeError("Cannot modify builder after output layer is added.")

    def _add_simple_rnn(
        self,
        hidden_dim,
        num_layers,
        nonlinearity,
        bidirectional,
        dropout
    ):

        self._check_not_built()
        rnn = nn.RNN(
            input_size=self.prev_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=self.batch_first,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.layers.append(rnn)
        self.layers.append(_RNNOutput())
        dirs = 2 if bidirectional else 1
        self.prev_dim = hidden_dim * dirs
        return self

    def _add_lstm(
        self,
        hidden_dim,
        num_layers,
        bidirectional,
        dropout
    ):

        self._check_not_built()
        lstm = nn.LSTM(
            input_size=self.prev_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=self.batch_first,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.layers.append(lstm)
        self.layers.append(_RNNOutput())
        dirs = 2 if bidirectional else 1
        self.prev_dim = hidden_dim * dirs
        return self

    def _add_gru(
        self,
        hidden_dim,
        num_layers,
        bidirectional,
        dropout
    ):

        self._check_not_built()
        gru = nn.GRU(
            input_size=self.prev_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=self.batch_first,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.layers.append(gru)
        self.layers.append(_RNNOutput())
        dirs = 2 if bidirectional else 1
        self.prev_dim = hidden_dim * dirs
        return self

    def _add_dropout(self, p):
        self._check_not_built()
        self.layers.append(nn.Dropout(p))
        return self

    def _add_output_layer(
        self,
        output_dim,
        activation
    ):

        self._check_not_built()

        if not self.seq_to_seq:
            self.layers.append(LastTimeStep(batch_first=self.batch_first))

        self.layers.append(nn.Linear(self.prev_dim, output_dim))
        if activation:
            self.layers.append(activation())
        model = nn.Sequential(*self.layers)
        self._built = True
        return model


def invoke(input_dim):
    builder = RecurrentNNBuilder(input_dim, {batch_first}, {seq_to_seq})
    return builder


meta = {
    "name": "RNN Input Layer",
    "min_args": 1,
    "max_args": 1
}

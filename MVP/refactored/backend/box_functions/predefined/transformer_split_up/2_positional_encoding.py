import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def invoke(builder):
    model_dim = builder.get_model_dim()
    dropout = builder.get_dropout()
    max_len = builder.get_max_len()
    builder.set_positional_encoding_layer(PositionalEncoding(model_dim, dropout, max_len))
    return builder


meta = {
    "name": "2 Positional Encoding",
    "min_args": 1,
    "max_args": 1
}

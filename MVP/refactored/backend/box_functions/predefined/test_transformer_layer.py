import torch.nn as nn


def invoke(model_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
    layer = nn.Transformer(d_model=model_dim,
                           nhead=num_heads,
                           num_encoder_layers=num_encoder_layers,
                           num_decoder_layers=num_decoder_layers,
                           dropout=dropout)
    return layer


meta = {
    "name": "Test Transformer Layer",
    "min_args": 4,
    "max_args": 5
}

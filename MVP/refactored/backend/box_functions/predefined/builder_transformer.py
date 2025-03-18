import torch.nn as nn
from torchinfo import summary
import torch
import math


def invoke(builder, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
    builder.set_transformer_layer(nn.Transformer(
        d_model=model_dim,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout)
    )
    return builder


meta = {
    "name": "Builder Transformer",
    "min_args": 5,
    "max_args": 6
}

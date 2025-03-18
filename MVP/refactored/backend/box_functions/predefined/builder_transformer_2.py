import torch.nn as nn
from torchinfo import summary
import torch
import math


def invoke(builder, model_dim, dropout=0.1):
    builder.set_transformer_layer(nn.Transformer(
        d_model=model_dim,
        nhead={num_heads},
        num_encoder_layers={num_encoder_layers},
        num_decoder_layers={num_decoder_layers},
        dropout=dropout)
    )
    return builder


meta = {
    "name": "Builder Transformer 2",
    "min_args": 2,
    "max_args": 3
}

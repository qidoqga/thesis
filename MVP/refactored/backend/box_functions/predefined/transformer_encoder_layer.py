import torch.nn as nn


def invoke(model, model_dim):
    # encoder_layer = nn.TransformerEncoderLayer({model_dim}, {num_heads}, dim_feedforward={dim_feedforward}, dropout={dropout})
    encoder_layer = nn.TransformerEncoderLayer(model_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
    model.add_module("encoder_layer", encoder_layer)
    return model


meta = {
    "name": "Transformer Encoder Layer",
    "min_args": 2,
    "max_args": 2
}

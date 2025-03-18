import torch.nn as nn


def invoke(model, model_dim):
    # decoder_layer = nn.TransformerDecoderLayer(
    #     d_model=model_dim,
    #     nhead=num_heads,
    #     dim_feedforward=dim_feedforward,
    #     dropout=dropout
    # )
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=model_dim,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1
    )
    model.add_module("decoder_layer", decoder_layer)
    return model


meta = {
    "name": "Transformer Decoder Layer",
    "min_args": 2,  # e.g. model and model_dim; you can supply others optionally.
    "max_args": 5   # model, model_dim, num_heads, dim_feedforward, dropout.
}

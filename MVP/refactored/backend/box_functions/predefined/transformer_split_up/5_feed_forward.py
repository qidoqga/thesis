import torch.nn as nn


def invoke(encoder_builder):
    builder = encoder_builder.get_builder()
    model_dim = builder.get_model_dim()
    d_ff = builder.get_d_ff()
    dropout = builder.get_dropout()

    encoder_builder.set_feed_forward(nn.Sequential(nn.Linear(model_dim, d_ff),
                                                   nn.ReLU(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(d_ff, model_dim)
                                                   ))
    return encoder_builder


meta = {
    "name": "5 Feed Forward",
    "min_args": 1,
    "max_args": 1
}

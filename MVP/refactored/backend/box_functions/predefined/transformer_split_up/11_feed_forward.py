import torch.nn as nn


def invoke(decoder_builder):
    builder = decoder_builder.get_builder()
    model_dim = builder.get_model_dim()
    d_ff = builder.get_d_ff()
    dropout = builder.get_dropout()

    decoder_builder.set_feed_forward(nn.Sequential(nn.Linear(model_dim, d_ff),
                                                   nn.ReLU(),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(d_ff, model_dim)
                                                   ))
    return decoder_builder


meta = {
    "name": "11 Feed Forward",
    "min_args": 1,
    "max_args": 1
}

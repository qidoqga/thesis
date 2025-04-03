import torch.nn as nn


def invoke(decoder_builder):
    builder = decoder_builder.get_builder()
    model_dim = builder.get_model_dim()

    decoder_builder.set_norm(nn.LayerNorm(model_dim))
    return decoder_builder


meta = {
    "name": "12 Normalization",
    "min_args": 1,
    "max_args": 1
}

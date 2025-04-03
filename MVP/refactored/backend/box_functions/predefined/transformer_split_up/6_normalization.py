import torch.nn as nn


def invoke(encoder_builder):
    builder = encoder_builder.get_builder()
    model_dim = builder.get_model_dim()

    encoder_builder.set_norm(nn.LayerNorm(model_dim))
    return encoder_builder


meta = {
    "name": "6 Normalization",
    "min_args": 1,
    "max_args": 1
}

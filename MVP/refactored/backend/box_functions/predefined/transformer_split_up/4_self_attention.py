import torch.nn as nn


def invoke(encoder_builder):
    builder = encoder_builder.get_builder()

    embed_dim = builder.get_model_dim()
    num_heads = builder.get_num_heads()
    dropout = builder.get_dropout()
    encoder_builder.set_self_attention_layer(nn.MultiheadAttention(embed_dim=embed_dim,
                                                                   num_heads=num_heads,
                                                                   dropout=dropout))
    return encoder_builder


meta = {
    "name": "4 Self Attention",
    "min_args": 1,
    "max_args": 1
}

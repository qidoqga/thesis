import torch.nn as nn


def invoke(decoder_builder):
    builder = decoder_builder.get_builder()
    embed_dim = builder.get_model_dim()
    num_heads = builder.get_num_heads()
    dropout = builder.get_dropout()

    decoder_builder.set_masked_self_attention(nn.MultiheadAttention(embed_dim=embed_dim,
                                                                    num_heads=num_heads,
                                                                    dropout=dropout))
    return decoder_builder


meta = {
    "name": "9 Masked Self Attention",
    "min_args": 1,
    "max_args": 1
}

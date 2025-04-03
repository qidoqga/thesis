

def invoke(builder):
    transformer_builder = builder.get_builder()

    model_dim = transformer_builder.get_model_dim()
    num_heads = transformer_builder.get_num_heads()
    dropout = transformer_builder.get_dropout()

    builder.add_block(ResidualWrapper(SelfAttentionBlock(model_dim, num_heads, dropout)))
    return builder


meta = {
    "name": "3b Self Attn",
    "min_args": 1,
    "max_args": 1
}

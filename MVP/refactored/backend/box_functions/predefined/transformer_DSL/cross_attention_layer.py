

def invoke(ca_builder, *args):
    transformer_builder = ca_builder.get_builder()

    d_model = transformer_builder.get_model_dim()
    nhead = transformer_builder.get_num_heads()
    dropout = transformer_builder.get_dropout()

    ca_block = ResidualWrapper(CrossAttentionBlock(d_model, nhead, dropout))

    ca_builder.add_block(ca_block)
    return ca_builder


meta = {
    "name": "Cross Attention Layer",
    "min_args": 1,
    "max_args": 2
}

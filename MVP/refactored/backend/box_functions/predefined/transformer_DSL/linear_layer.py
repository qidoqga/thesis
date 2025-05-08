

def invoke(builder):
    transformer_builder = builder.get_builder()
    model_dim = transformer_builder.get_model_dim()
    d_ff = transformer_builder.get_d_ff()

    if not builder.blocks:
        builder.add_block(LinearBlock(model_dim, d_ff))
    else:
        last_box = next(
            (block for block in reversed(builder.blocks) if isinstance(block, (LinearBlock, NormBlock))),
            None
        )
        out_features = last_box.out_features
        builder.add_block(LinearBlock(out_features, d_ff))

    return builder


meta = {
    "name": "Linear Layer",
    "min_args": 1,
    "max_args": 1
}



def invoke(builder):
    if not builder.blocks:
        transformer_builder = builder.get_builder()
        model_dim = transformer_builder.get_model_dim()
        builder.add_block(NormBlock(model_dim))

    else:
        last_box = next(
            (block for block in reversed(builder.blocks) if isinstance(block, LinearBlock)),
            None
        )
        out_features = last_box.out_features
        builder.add_block(NormBlock(out_features))
    return builder


meta = {
    "name": "3a Norm",
    "min_args": 1,
    "max_args": 1
}

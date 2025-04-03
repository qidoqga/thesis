

def invoke(ff_builder):
    builder = ff_builder.builder
    last_box = next(
        (block for block in reversed(ff_builder.blocks) if isinstance(block, LinearBlock)),
        None
    )
    last_box.set_out_features(builder.model_dim)

    ff_pipeline = ResidualWrapper(ff_builder.build())


    encoder_builder = builder.encoder_builder
    encoder_builder.set_feed_forward_pipeline(ff_pipeline)

    builder.set_encoder_builder(encoder_builder)

    encoder_layer = encoder_builder.build()
    builder.encoder_stack.append(encoder_layer)
    return builder


meta = {
    "name": "4d Build FF",
    "min_args": 1,
    "max_args": 1
}



def invoke(ff_builder):
    builder = ff_builder.builder
    last_box = next(
        (block for block in reversed(ff_builder.blocks) if isinstance(block, LinearBlock)),
        None
    )
    last_box.set_out_features(builder.model_dim)

    ff_pipeline = ResidualWrapper(ff_builder.build())

    decoder_builder = builder.decoder_builder
    decoder_builder.set_feed_forward_pipeline(ff_pipeline)

    builder.set_decoder_builder(decoder_builder)

    decoder_layer = decoder_builder.build()
    builder.decoder_stack.append(decoder_layer)
    return builder


meta = {
    "name": "Build Decoder",
    "min_args": 1,
    "max_args": 1
}

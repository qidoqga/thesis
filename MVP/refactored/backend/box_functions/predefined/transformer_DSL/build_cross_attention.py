

def invoke(ca_builder):
    cross_attn_pipeline = ca_builder.build()

    builder = ca_builder.builder

    decoder_builder = builder.decoder_builder
    decoder_builder.set_cross_attention_pipeline(cross_attn_pipeline)

    builder.set_decoder_builder(decoder_builder)
    return builder


meta = {
    "name": "Build Cross Attention",
    "min_args": 1,
    "max_args": 1
}

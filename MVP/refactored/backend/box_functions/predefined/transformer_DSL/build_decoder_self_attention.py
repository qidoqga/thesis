

def invoke(sa_builder):
    self_attn_pipeline = sa_builder.build()

    builder = sa_builder.builder

    decoder_builder = builder.decoder_builder
    decoder_builder.set_self_attention_pipeline(self_attn_pipeline)

    builder.set_decoder_builder(decoder_builder)
    return builder


meta = {
    "name": "Build Decoder Self Attention",
    "min_args": 1,
    "max_args": 1
}

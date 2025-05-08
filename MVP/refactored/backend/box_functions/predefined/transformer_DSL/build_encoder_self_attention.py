

def invoke(sa_builder):
    self_attn_pipeline = sa_builder.build()

    builder = sa_builder.get_builder()

    encoder_builder = builder.get_encoder_builder()
    encoder_builder.set_self_attention_pipeline(self_attn_pipeline)

    builder.set_encoder_builder(encoder_builder)
    return builder


meta = {
    "name": "Build Encoder Self Attention",
    "min_args": 1,
    "max_args": 1
}

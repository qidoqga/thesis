

def invoke(builder):
    ca_builder = CrossAttentionPipelineBuilder(builder)
    return ca_builder


meta = {
    "name": "10 Cross Attn Builder",
    "min_args": 1,
    "max_args": 1
}



def invoke(builder):
    ff_builder = FeedForwardPipelineBuilder(builder)
    return ff_builder


meta = {
    "name": "6 FF Builder",
    "min_args": 1,
    "max_args": 1
}

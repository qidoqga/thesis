

def invoke(builder):
    ff_builder = FeedForwardPipelineBuilder(builder)
    return ff_builder


meta = {
    "name": "4 FF Builder",
    "min_args": 1,
    "max_args": 1
}

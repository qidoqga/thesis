
def invoke(builder):
    builder._add_batch_norm()
    return builder


meta = {
    "name": "FFN Batch Norm",
    "min_args": 1,
    "max_args": 1
}

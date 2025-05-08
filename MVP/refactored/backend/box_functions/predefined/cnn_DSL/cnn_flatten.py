
def invoke(builder):
    builder._add_flatten()
    return builder


meta = {
    "name": "CNN Flatten",
    "min_args": 1,
    "max_args": 1
}

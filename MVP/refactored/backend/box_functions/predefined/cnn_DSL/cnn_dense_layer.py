
def invoke(builder):
    builder._add_dense({neurons}, {activation})
    return builder


meta = {
    "name": "CNN Dense Layer",
    "min_args": 1,
    "max_args": 1
}

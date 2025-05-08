
def invoke(builder):
    builder._add_hidden_layer({neurons}, activation={activation})
    return builder


meta = {
    "name": "FFN Hidden Layer",
    "min_args": 1,
    "max_args": 2
}


def invoke(builder):
    return builder._add_output_layer({neurons}, {activation})


meta = {
    "name": "RNN Output Layer",
    "min_args": 1,
    "max_args": 1
}

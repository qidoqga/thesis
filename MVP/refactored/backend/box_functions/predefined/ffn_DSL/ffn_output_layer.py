
def invoke(builder):
    return builder._add_output_layer({outputs}, activation={activation})


meta = {
    "name": "FFN Output Layer",
    "min_args": 1,
    "max_args": 1
}


def invoke(builder):
    builder._add_conv({out_channels}, {kernel_size}, {stride}, {padding}, {activation})
    return builder


meta = {
    "name": "CNN Conv Layer",
    "min_args": 1,
    "max_args": 1
}

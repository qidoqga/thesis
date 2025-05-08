
def invoke(builder):
    builder._add_pool({kernel_size}, {stride}, {pool_type})
    return builder


meta = {
    "name": "CNN Pool",
    "min_args": 1,
    "max_args": 1
}

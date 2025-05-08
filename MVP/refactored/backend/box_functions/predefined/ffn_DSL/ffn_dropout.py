
def invoke(builder):
    builder._add_dropout({dropout})
    return builder


meta = {
    "name": "FFN Dropout",
    "min_args": 1,
    "max_args": 1
}

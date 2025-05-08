
def invoke(builder):
    builder._add_dropout({dropout})
    return builder


meta = {
    "name": "RNN Dropout",
    "min_args": 1,
    "max_args": 1
}

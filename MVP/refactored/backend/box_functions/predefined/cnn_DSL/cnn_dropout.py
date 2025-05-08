
def invoke(builder):
    builder._add_dropout({dropout})
    return builder


meta = {
    "name": "CNN Dropout",
    "min_args": 1,
    "max_args": 1
}

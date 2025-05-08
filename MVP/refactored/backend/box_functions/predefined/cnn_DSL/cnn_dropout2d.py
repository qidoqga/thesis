
def invoke(builder):
    builder._add_dropout2d({dropout})
    return builder


meta = {
    "name": "CNN Dropout2d",
    "min_args": 1,
    "max_args": 1
}
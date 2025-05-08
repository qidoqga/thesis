
def invoke(builder):
    builder._add_lstm({neurons}, {num_layers}, {bidirectional}, {dropout})
    return builder


meta = {
    "name": "RNN Lstm Layer",
    "min_args": 1,
    "max_args": 1
}

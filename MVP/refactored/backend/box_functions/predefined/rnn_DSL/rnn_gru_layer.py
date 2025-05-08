
def invoke(builder):
    builder._add_gru({neurons}, {num_layers}, {bidirectional}, {dropout})
    return builder


meta = {
    "name": "RNN Gru Layer",
    "min_args": 1,
    "max_args": 1
}

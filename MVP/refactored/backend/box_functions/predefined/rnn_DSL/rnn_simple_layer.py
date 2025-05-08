
def invoke(builder):
    builder._add_simple_rnn({neurons}, {num_layers}, {non_linearity}, {bidirectional}, {dropout})
    return builder


meta = {
    "name": "RNN Simple Layer",
    "min_args": 1,
    "max_args": 1
}

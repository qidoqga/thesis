

def invoke(builder):
    transformer_builder = builder.builder

    activation = transformer_builder.activation

    builder.add_block(ActivationBlock(activation))
    return builder


meta = {
    "name": "Activation Layer",
    "min_args": 1,
    "max_args": 1
}

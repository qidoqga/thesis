

def invoke(builder):
    transformer_builder = builder.builder

    activation = transformer_builder.activation_in_encoder

    builder.add_block(ActivationBlock(activation))
    return builder


meta = {
    "name": "4b Act",
    "min_args": 1,
    "max_args": 1
}

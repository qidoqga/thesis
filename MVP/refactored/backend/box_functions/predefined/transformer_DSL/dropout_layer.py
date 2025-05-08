

def invoke(builder):
    transformer_builder = builder.builder
    dropout = transformer_builder.dropout

    builder.add_block(DropoutBlock(dropout))
    return builder


meta = {
    "name": "Dropout Layer",
    "min_args": 1,
    "max_args": 1
}

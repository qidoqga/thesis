import torch.nn as nn


def invoke(builder):
    vocab_size = builder.get_vocab_size()
    model_dim = builder.get_model_dim()
    builder.set_embedding_layer(nn.Embedding(vocab_size, model_dim))
    return builder


meta = {
    "name": "1 Embedd",
    "min_args": 1,
    "max_args": 1
}

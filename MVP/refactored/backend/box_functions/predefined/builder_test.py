import torch.nn as nn
from torchinfo import summary
import torch
import math


class TransformerModelBuilder:
    def __init__(self, model_dim):
        self.model_dim = model_dim
        self.embedding_layer = None
        self.positional_encoding_layer = None
        self.transformer_layer = None
        self.linear_layer = None

    def set_embedding_layer(self, embedding_layer):
        self.embedding_layer = embedding_layer
        return self

    def set_positional_encoding_layer(self, positional_encoding_layer):
        self.positional_encoding_layer = positional_encoding_layer
        return self

    def set_transformer_layer(self, transformer_layer):
        self.transformer_layer = transformer_layer
        return self

    def set_linear_layer(self, linear_layer):
        self.linear_layer = linear_layer
        return self

    def build(self):
        if (self.embedding_layer is None or
                self.positional_encoding_layer is None or
                self.transformer_layer is None or
                self.linear_layer is None):
            raise ValueError("Not all layers have been set.")

        return TransformerModel(
            self.model_dim,
            self.embedding_layer,
            self.positional_encoding_layer,
            self.transformer_layer,
            self.linear_layer
        )


def invoke(model_dim):
    return TransformerModelBuilder(model_dim)


meta = {
    "name": "Builder Test",
    "min_args": 1,
    "max_args": 1
}

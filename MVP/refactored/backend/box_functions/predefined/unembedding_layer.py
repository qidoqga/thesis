import torch.nn as nn


class UnembeddingLayer(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(UnembeddingLayer, self).__init__()
        self.fc = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)


def invoke(vocab_size, model_dim):
    layer = UnembeddingLayer(model_dim, vocab_size)
    return layer


meta = {
    "name": "Unembedding Layer",
    "min_args": 2,
    "max_args": 2
}

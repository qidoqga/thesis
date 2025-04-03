import torch.nn as nn


class OutputHead(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(OutputHead, self).__init__()
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        return self.fc_out(x)


def invoke(builder):
    model_dim = builder.model_dim
    vocab_size = builder.vocab_size

    builder.set_output_head(OutputHead(model_dim, vocab_size))
    return builder


meta = {
    "name": "7 Output Head",
    "min_args": 1,
    "max_args": 1
}

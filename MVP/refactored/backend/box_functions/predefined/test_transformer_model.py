import torch.nn as nn
import math


class TransformerModel(nn.Module):
    def __init__(self, model_dim, embedding_layer, positional_encoding_layer, transformer_layer, linear_layer):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim

        self.embedding = embedding_layer

        self.pos_encoder = positional_encoding_layer

        self.transformer = transformer_layer

        self.fc_out = linear_layer

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (src_seq_len, batch_size)
        tgt: (tgt_seq_len, batch_size)
        """
        src_emb = self.embedding(src) * math.sqrt(self.model_dim)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.model_dim)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)

        output = self.fc_out(output)
        output = self.softmax(output)
        return output


def invoke(model_dim, embedding_layer, positional_encoding_layer, transformer_layer, linear_layer):
    model = TransformerModel(model_dim, embedding_layer, positional_encoding_layer, transformer_layer, linear_layer)
    return model


meta = {
    "name": "Test Transformer Model",
    "min_args": 5,
    "max_args": 5
}

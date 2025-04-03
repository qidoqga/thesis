import torch.nn as nn
import math


class CustomTransformer(nn.Module):
    def __init__(self, model_dim, embedding_layer, positional_encoding_layer, encoder_stack, decoder_stack, linear_layer):
        super(CustomTransformer, self).__init__()
        self.model_dim = model_dim

        self.embedding = embedding_layer

        self.pos_encoder = positional_encoding_layer

        self.encoder_stack = encoder_stack

        self.decoder_stack = decoder_stack

        self.fc_out = linear_layer

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.model_dim)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.model_dim)
        tgt_emb = self.pos_encoder(tgt_emb)

        encoder_output = self.encoder_stack(src_emb)
        decoder_output = self.decoder_stack(tgt_emb, encoder_output, tgt_mask=tgt_mask)

        output = self.fc_out(decoder_output)
        output = self.softmax(output)
        return output


def invoke(builder):
    model = builder.build()
    return model


meta = {
    "name": "15 Transformer Model Builder",
    "min_args": 1,
    "max_args": 1
}

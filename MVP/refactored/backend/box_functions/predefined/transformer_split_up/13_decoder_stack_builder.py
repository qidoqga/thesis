import torch.nn as nn


class CustomDecoderLayer(nn.Module):
    def __init__(self, masked_self_attn, self_attn_norm, cross_attn, cross_attn_norm, ffn, ffn_norm, dropout=0.1):
        super(CustomDecoderLayer, self).__init__()
        self.masked_self_attn = masked_self_attn  # masked self-attention module
        self.self_attn_norm = self_attn_norm  # layer norm after self-attention
        self.cross_attn = cross_attn  # cross-attention module
        self.cross_attn_norm = cross_attn_norm  # layer norm after cross-attention
        self.ffn = ffn  # feed-forward network
        self.ffn_norm = ffn_norm  # layer norm after feed-forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        # 1. Masked self-attention on target (tgt)
        tgt2, _ = self.masked_self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.self_attn_norm(tgt + self.dropout(tgt2))

        # 2. Cross-attention: tgt attends to encoder output
        tgt2, _ = self.cross_attn(tgt, encoder_output, encoder_output, attn_mask=memory_mask)
        tgt = self.cross_attn_norm(tgt + self.dropout(tgt2))

        # 3. Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = self.ffn_norm(tgt + self.dropout(tgt2))

        return tgt


class CustomDecoderStack(nn.Module):
    def __init__(self, layers):
        super(CustomDecoderStack, self).__init__()
        self.layers = layers

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output


class DecoderStackBuilder:
    def __init__(self, decoder_layer_builder):
        self.decoder_layer_builder = decoder_layer_builder
        self.builder = decoder_layer_builder.get_builder()
        self.num_layers = self.builder.get_num_layers()

    def build_stack(self):
        layers = nn.ModuleList([self.decoder_layer_builder.build_layer() for _ in range(self.num_layers)])
        return CustomDecoderStack(layers)


def invoke(decoder_builder):
    decoder_stack_builder = DecoderStackBuilder(decoder_builder)
    decoder_stack = decoder_stack_builder.build_stack()

    builder = decoder_builder.get_builder()
    builder.set_decoder_stack(decoder_stack)
    return builder


meta = {
    "name": "13 Decoder Stack Builder",
    "min_args": 1,
    "max_args": 1
}

import torch.nn as nn


class EncoderStackBuilder:
    def __init__(self, encoder_layer_builder):
        self.encoder_layer_builder = encoder_layer_builder
        self.builder = encoder_layer_builder.get_builder()
        self.num_layers = self.builder.get_num_layers()

    def build_stack(self):
        layers = [self.encoder_layer_builder.build_layer() for _ in range(self.num_layers)]
        return nn.Sequential(*layers)


class CustomEncoderLayer(nn.Module):
    def __init__(self, self_attn, attn_norm, ffn, ffn_norm, dropout=0.1):
        super(CustomEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.attn_norm = attn_norm
        self.ffn = ffn
        self.ffn_norm = ffn_norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.attn_norm(src + self.dropout(src2))

        src2 = self.ffn(src)
        src = self.ffn_norm(src + self.dropout(src2))

        return src


def invoke(encoder_builder):
    encoder_stack_builder = EncoderStackBuilder(encoder_builder)
    encoder_stack = encoder_stack_builder.build_stack()

    builder = encoder_builder.get_builder()
    builder.set_encoder_stack(encoder_stack)

    return builder


meta = {
    "name": "7 Encoder Stack Builder",
    "min_args": 1,
    "max_args": 1
}


class DecoderLayerBuilder:
    def __init__(self, builder):
        """
        model_dim: dimension of embeddings/hidden states.
        num_heads: number of attention heads.
        d_ff: dimension of the feed-forward network.
        dropout: dropout probability.
        """
        self.builder = builder
        # self.model_dim = model_dim
        # self.num_heads = num_heads
        # self.d_ff = d_ff
        self.dropout = builder.get_dropout()

        self.masked_self_attention = None
        self.cross_attention = None
        self.feed_forward = None
        self.norm = None

    def set_masked_self_attention(self, masked_self_attention):
        self.masked_self_attention = masked_self_attention
        return self

    def set_cross_attention(self, cross_attention):
        self.cross_attention = cross_attention
        return self

    def set_feed_forward(self, ff_layer):
        self.feed_forward = ff_layer
        return self

    def set_norm(self, norm):
        self.norm = norm
        return self

    def get_builder(self):
        return self.builder

    def build_layer(self):
        return CustomDecoderLayer(
            masked_self_attn=self.masked_self_attention,
            self_attn_norm=self.norm,
            cross_attn=self.cross_attention,
            cross_attn_norm=self.norm,
            ffn=self.feed_forward,
            ffn_norm=self.norm,
            dropout=self.dropout
        )


def invoke(builder):
    return DecoderLayerBuilder(builder)


meta = {
    "name": "8 Decoder Builder",
    "min_args": 1,
    "max_args": 1
}

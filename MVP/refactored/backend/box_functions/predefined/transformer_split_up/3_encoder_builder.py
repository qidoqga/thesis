
class EncoderLayerBuilder:
    def __init__(self, builder):
        self.builder = builder

        # self.model_dim = builder.get_model_dim()
        # self.num_heads = builder.get_num_heads()
        # self.d_ff = builder.get_d_ff()
        self.dropout = builder.get_dropout()

        self.self_attention = None
        self.feed_forward = None
        self.norm = None

    def set_self_attention_layer(self, attention_layer):
        self.self_attention = attention_layer
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
        return CustomEncoderLayer(
            self.self_attention,
            self.norm,
            self.feed_forward,
            self.norm,
            self.dropout
        )


def invoke(builder):
    return EncoderLayerBuilder(builder)


meta = {
    "name": "3 Encoder Builder",
    "min_args": 1,
    "max_args": 1
}

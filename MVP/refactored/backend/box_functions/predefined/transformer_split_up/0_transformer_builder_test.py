
class TransformerModelBuilder:
    def __init__(self, model_dim, vocab_size, num_heads=8, num_layers=3, dropout=0.1, max_len=500, d_ff=2048):
        self.model_dim = model_dim
        self.embedding_layer = None
        self.positional_encoding_layer = None
        self.linear_layer = None

        self.encoder_stack = None
        self.decoder_stack = None
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.d_ff = d_ff

    def set_embedding_layer(self, embedding_layer):
        self.embedding_layer = embedding_layer
        return self

    def set_positional_encoding_layer(self, positional_encoding_layer):
        self.positional_encoding_layer = positional_encoding_layer
        return self

    def set_linear_layer(self, linear_layer):
        self.linear_layer = linear_layer
        return self

    def set_encoder_stack(self, encoder_stack):
        self.encoder_stack = encoder_stack
        return self

    def set_decoder_stack(self, decoder_stack):
        self.decoder_stack = decoder_stack
        return self

    def get_model_dim(self):
        return self.model_dim

    def get_vocab_size(self):
        return self.vocab_size

    def get_num_heads(self):
        return self.num_heads

    def get_num_layers(self):
        return self.num_layers

    def get_dropout(self):
        return self.dropout

    def get_max_len(self):
        return self.max_len

    def get_d_ff(self):
        return self.d_ff

    def build(self):
        if (self.embedding_layer is None or
                self.positional_encoding_layer is None or
                self.linear_layer is None or
                self.encoder_stack is None or
                self.decoder_stack is None):
            raise ValueError("Not all layers have been set.")

        return CustomTransformer(
            self.model_dim,
            self.embedding_layer,
            self.positional_encoding_layer,
            self.encoder_stack,
            self.decoder_stack,
            self.linear_layer
        )


def invoke(model_dim, vocab_size, num_heads=8, num_layers=3, dropout=0.1, max_len=500, d_ff=2048):
    return TransformerModelBuilder(model_dim, vocab_size, num_heads, num_layers, dropout, max_len, d_ff)


meta = {
    "name": "0 Transformer Builder Test",
    "min_args": 2,
    "max_args": 7
}

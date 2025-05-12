import torch
import torch.nn as nn


class TransformerModelBuilder:
    def __init__(self, model_dim, vocab_size, num_heads=8, activation='relu', dropout=0.1, max_len=500, d_ff=2048):
        self.model_dim = model_dim
        self.embedding_layer = None
        self.positional_encoding_layer = None
        # self.linear_layer = None
        self.output_head = None

        self.encoder_stack = []
        self.decoder_stack = []
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_len = max_len
        self.d_ff = d_ff

        self.encoder_builder = None
        self.decoder_builder = None
        self.activation = activation
        self.activation_in_encoder = "relu"
        self.activation_in_decoder = "relu"

    def set_embedding_layer(self, embedding_layer):
        self.embedding_layer = embedding_layer
        return self

    def set_positional_encoding_layer(self, positional_encoding_layer):
        self.positional_encoding_layer = positional_encoding_layer
        return self

    # def set_linear_layer(self, linear_layer):
    #     self.linear_layer = linear_layer
    #     return self

    def set_output_head(self, output_head):
        self.output_head = output_head
        return self

    def set_encoder_stack(self, encoder_stack):
        self.encoder_stack = encoder_stack
        return self

    def set_decoder_stack(self, decoder_stack):
        self.decoder_stack = decoder_stack
        return self

    def set_encoder_builder(self, encoder_builder):
        self.encoder_builder = encoder_builder
        return self

    def set_decoder_builder(self, decoder_builder):
        self.decoder_builder = decoder_builder
        return self

    def get_model_dim(self):
        return self.model_dim

    def get_vocab_size(self):
        return self.vocab_size

    def get_num_heads(self):
        return self.num_heads

    def get_dropout(self):
        return self.dropout

    def get_max_len(self):
        return self.max_len

    def get_d_ff(self):
        return self.d_ff

    def get_encoder_builder(self):
        return self.encoder_builder

    def build_model(self):
        return TransformerModel(
            embedding_layer=self.embedding_layer,
            positional_encoding_layer=self.positional_encoding_layer,
            encoder_stack=self.encoder_stack,  # a list of encoder layers
            decoder_stack=self.decoder_stack,  # a list of decoder layers
            output_head=self.output_head
        )


class NormBlock(nn.Module):
    """Applies a LayerNorm to the input."""

    def __init__(self, d_model):
        super(NormBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.out_features = d_model

    def forward(self, x, **kwargs):
        return self.norm(x)


class DropoutBlock(nn.Module):
    """Applies dropout to the input."""

    def __init__(self, dropout):
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        return self.dropout(x)


class LinearBlock(nn.Module):
    """A linear layer."""

    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, **kwargs):
        return self.linear(x)

    def get_in_features(self):
        return self.in_features

    def get_out_features(self):
        return self.out_features

    def set_out_features(self, out_features):
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)


class ActivationBlock(nn.Module):
    """An activation function block."""

    def __init__(self, activation="relu"):
        super(ActivationBlock, self).__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, **kwargs):
        return self.act(x)


class SelfAttentionBlock(nn.Module):
    """A self-attention block that returns its output (without residual)."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, **kwargs):
        # Expect kwargs to contain 'mask' and 'key_padding_mask' if needed.
        mask = kwargs.get("mask", None)
        key_padding_mask = kwargs.get("key_padding_mask", None)
        # MultiheadAttention expects inputs of shape (seq_len, batch_size, d_model)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=key_padding_mask)
        return attn_out


class ResidualWrapper(nn.Module):
    """Wraps a block with a residual connection."""

    def __init__(self, block):
        super(ResidualWrapper, self).__init__()
        self.block = block

    def forward(self, x, **kwargs):
        return x + self.block(x, **kwargs)


class Pipeline(nn.Module):
    """Applies a list of blocks sequentially. Each block must accept (x, **kwargs)."""

    def __init__(self, blocks):
        super(Pipeline, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, **kwargs):
        for block in self.blocks:
            x = block(x, **kwargs)
        return x


class SelfAttentionPipelineBuilder:
    """
    Builder for a self-attention pipeline.
    You can add blocks manually. For example, you might add:
      Norm -> SelfAttention wrapped in Residual -> Optional Dropout
    """

    def __init__(self, builder):
        self.builder = builder
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)
        return self  # For chaining

    def build(self):
        return Pipeline(self.blocks)

    def get_builder(self):
        return self.builder


class FeedForwardPipelineBuilder:
    """
    Builder for a feed-forward pipeline.
    For instance, you can add blocks such as:
      Norm -> Linear -> Activation -> Dropout -> Linear -> (optional Dropout)
    And then wrap the whole pipeline in a residual connection.
    """

    def __init__(self, builder):
        self.builder = builder
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)
        return self

    def build(self):
        return Pipeline(self.blocks)

    def get_builder(self):
        return self.builder


def invoke(model_dim, vocab_size, num_heads=8, activation='relu', dropout=0.1, max_len=500, d_ff=2048):
    return TransformerModelBuilder(model_dim, vocab_size, num_heads, activation, dropout, max_len, d_ff)


meta = {
    "name": "Transformer Builder",
    "min_args": 2,
    "max_args": 7
}

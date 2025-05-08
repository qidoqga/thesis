import torch
import torch.nn as nn
import math


# Basic Block Definitions
# class NormBlock(nn.Module):
#     """Applies a LayerNorm to the input."""
#
#     def __init__(self, d_model):
#         super(NormBlock, self).__init__()
#         self.norm = nn.LayerNorm(d_model)
#         self.out_features = d_model
#
#     def forward(self, x, **kwargs):
#         return self.norm(x)
#
#
# class DropoutBlock(nn.Module):
#     """Applies dropout to the input."""
#
#     def __init__(self, dropout):
#         super(DropoutBlock, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, **kwargs):
#         return self.dropout(x)
#
#
# class LinearBlock(nn.Module):
#     """A linear layer."""
#
#     def __init__(self, in_features, out_features):
#         super(LinearBlock, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.in_features = in_features
#         self.out_features = out_features
#
#     def forward(self, x, **kwargs):
#         return self.linear(x)
#
#     def get_in_features(self):
#         return self.in_features
#
#     def get_out_features(self):
#         return self.out_features
#
#     def set_out_features(self, out_features):
#         self.out_features = out_features
#         self.linear = nn.Linear(self.in_features, self.out_features)
#
#
# class ActivationBlock(nn.Module):
#     """An activation function block."""
#
#     def __init__(self, activation="relu"):
#         super(ActivationBlock, self).__init__()
#         if activation == "relu":
#             self.act = nn.ReLU()
#         elif activation == "gelu":
#             self.act = nn.GELU()
#         else:
#             raise ValueError(f"Unsupported activation: {activation}")
#
#     def forward(self, x, **kwargs):
#         return self.act(x)
#
#
# class SelfAttentionBlock(nn.Module):
#     """A self-attention block that returns its output (without residual)."""
#
#     def __init__(self, d_model, nhead, dropout=0.1):
#         super(SelfAttentionBlock, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#
#     def forward(self, x, **kwargs):
#         # Expect kwargs to contain 'mask' and 'key_padding_mask' if needed.
#         mask = kwargs.get("mask", None)
#         key_padding_mask = kwargs.get("key_padding_mask", None)
#         # MultiheadAttention expects inputs of shape (seq_len, batch_size, d_model)
#         attn_out, _ = self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=key_padding_mask)
#         return attn_out
#
#
# class ResidualWrapper(nn.Module):
#     """Wraps a block with a residual connection."""
#
#     def __init__(self, block):
#         super(ResidualWrapper, self).__init__()
#         self.block = block
#
#     def forward(self, x, **kwargs):
#         return x + self.block(x, **kwargs)
#
#
# # Pipeline Module
# class Pipeline(nn.Module):
#     """Applies a list of blocks sequentially. Each block must accept (x, **kwargs)."""
#
#     def __init__(self, blocks):
#         super(Pipeline, self).__init__()
#         self.blocks = nn.ModuleList(blocks)
#
#     def forward(self, x, **kwargs):
#         for block in self.blocks:
#             x = block(x, **kwargs)
#         return x
#
#
# # Builders for Self-Attention and Feed-Forward Pipelines
# class SelfAttentionPipelineBuilder:
#     """
#     Builder for a self-attention pipeline.
#     You can add blocks manually. For example, you might add:
#       Norm -> SelfAttention wrapped in Residual -> Optional Dropout
#     """
#
#     def __init__(self, builder):
#         self.builder = builder
#         self.blocks = []
#
#     def add_block(self, block):
#         self.blocks.append(block)
#         return self  # For chaining
#
#     def build(self):
#         return Pipeline(self.blocks)
#
#     def get_builder(self):
#         return self.builder
#
#
# class FeedForwardPipelineBuilder:
#     """
#     Builder for a feed-forward pipeline.
#     For instance, you can add blocks such as:
#       Norm -> Linear -> Activation -> Dropout -> Linear -> (optional Dropout)
#     And then wrap the whole pipeline in a residual connection.
#     """
#
#     def __init__(self, builder):
#         self.builder = builder
#         self.blocks = []
#
#     def add_block(self, block):
#         self.blocks.append(block)
#         return self
#
#     def build(self):
#         return Pipeline(self.blocks)
#
#     def get_builder(self):
#         return self.builder


# Custom Encoder Layer Builder
class CustomEncoderLayer(nn.Module):
    """
    A custom encoder layer that applies a self-attention pipeline followed by a feed-forward pipeline.
    """

    def __init__(self, self_attn_pipeline, ff_pipeline):
        super(CustomEncoderLayer, self).__init__()
        self.self_attn_pipeline = self_attn_pipeline
        self.ff_pipeline = ff_pipeline

    def forward(self, x, mask=None, key_padding_mask=None):
        # Pass mask parameters to self-attention pipeline blocks.
        x = self.self_attn_pipeline(x, mask=mask, key_padding_mask=key_padding_mask)
        # For feed-forward pipeline, additional kwargs are not needed.
        x = self.ff_pipeline(x)
        return x


class EncoderLayerBuilder:
    """
    Builder for a custom encoder layer.
    You add a self-attention pipeline and a feed-forward pipeline.
    """

    def __init__(self):
        self.self_attn_pipeline = None
        self.ff_pipeline = None

    def set_self_attention_pipeline(self, pipeline):
        self.self_attn_pipeline = pipeline
        return self

    def set_feed_forward_pipeline(self, pipeline):
        self.ff_pipeline = pipeline
        return self

    def build(self):
        if self.self_attn_pipeline is None or self.ff_pipeline is None:
            raise ValueError("Both self-attention and feed-forward pipelines must be provided.")
        return CustomEncoderLayer(self.self_attn_pipeline, self.ff_pipeline)


def invoke(builder):
    builder.set_encoder_builder(EncoderLayerBuilder())
    sa_builder = SelfAttentionPipelineBuilder(builder)
    return sa_builder


meta = {
    "name": "3 Enc Builder",
    "min_args": 1,
    "max_args": 1
}

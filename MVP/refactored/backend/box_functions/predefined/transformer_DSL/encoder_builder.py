import torch
import torch.nn as nn
import math


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
    "name": "Encoder Builder",
    "min_args": 1,
    "max_args": 1
}

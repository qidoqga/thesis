import torch.nn as nn


# Custom Decoder Layer Builder
class CustomDecoderLayer(nn.Module):
    """
    A custom decoder layer that applies a masked self-attention pipeline
    followed by a feed-forward pipeline.

    The self-attention pipeline expects mask parameters (tgt_mask and
    tgt_key_padding_mask) passed as 'mask' and 'key_padding_mask'.
    """

    def __init__(self, self_attn_pipeline, ff_pipeline):
        super(CustomDecoderLayer, self).__init__()
        self.self_attn_pipeline = self_attn_pipeline
        self.ff_pipeline = ff_pipeline

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # Masked self-attention with residual connection and pre-norm:
        # Pass the target mask parameters to the self-attention pipeline.
        tgt = self.self_attn_pipeline(tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        # Feed-forward pipeline with residual connection and pre-norm.
        tgt = self.ff_pipeline(tgt)
        return tgt


class DecoderLayerBuilder:
    """
    Builder for a custom decoder layer.
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
        return CustomDecoderLayer(self.self_attn_pipeline, self.ff_pipeline)


def invoke(builder):
    builder.set_decoder_builder(DecoderLayerBuilder())
    sa_builder = SelfAttentionPipelineBuilder(builder)
    return sa_builder


meta = {
    "name": "5 Dec Builder",
    "min_args": 1,
    "max_args": 1
}

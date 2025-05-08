import torch.nn as nn


# Custom Decoder Layer Builder
class CustomDecoderLayer(nn.Module):
    """
    A custom decoder layer that applies a masked self-attention pipeline
    followed by a feed-forward pipeline.

    The self-attention pipeline expects mask parameters (tgt_mask and
    tgt_key_padding_mask) passed as 'mask' and 'key_padding_mask'.
    """

    def __init__(self, self_attn_pipeline, cross_attn_pipeline, ff_pipeline):
        super(CustomDecoderLayer, self).__init__()
        self.self_attn_pipeline = self_attn_pipeline
        self.cross_attn_pipeline = cross_attn_pipeline
        self.ff_pipeline = ff_pipeline

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):

        x = self.self_attn_pipeline(
            tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )

        if self.cross_attn_pipeline is not None:
            x2 = self.cross_attn_pipeline(
                x,
                mask=None,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask
            )
            x = x + x2

        return self.ff_pipeline(x)


class DecoderLayerBuilder:
    """
    Builder for a custom decoder layer.
    You add a self-attention pipeline and a feed-forward pipeline.
    """

    def __init__(self):
        self.self_attn_pipeline = None
        self.cross_attention_pipeline = None
        self.ff_pipeline = None

    def set_self_attention_pipeline(self, pipeline):
        self.self_attn_pipeline = pipeline
        return self

    def set_cross_attention_pipeline(self, pipeline):
        self.cross_attention_pipeline = pipeline

    def set_feed_forward_pipeline(self, pipeline):
        self.ff_pipeline = pipeline
        return self

    def build(self):
        if self.self_attn_pipeline is None or self.ff_pipeline is None:
            raise ValueError("Both self-attention and feed-forward pipelines must be provided.")
        return CustomDecoderLayer(self.self_attn_pipeline, self.cross_attention_pipeline, self.ff_pipeline)


class CrossAttentionBlock(nn.Module):
    """A cross‑attention block: queries from decoder, keys/values from encoder memory."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, memory, **kwargs):
        # x:      (tgt_len, batch, d_model)
        # memory: (src_len, batch, d_model)
        mask = kwargs.get("mask", None)
        key_padding_mask = kwargs.get("key_padding_mask", None)
        memory_key_padding_mask = kwargs.get("memory_key_padding_mask", None)
        attn_out, _ = self.cross_attn(
            x, memory, memory,
            attn_mask=mask,
            key_padding_mask=memory_key_padding_mask
        )
        return attn_out


class CrossAttentionPipelineBuilder:
    """
    Builder for a cross‑attention pipeline in decoder:
      Norm -> Residual(CrossAttention) -> Dropout
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


def invoke(builder):
    builder.set_decoder_builder(DecoderLayerBuilder())
    sa_builder = SelfAttentionPipelineBuilder(builder)
    return sa_builder


meta = {
    "name": "Decoder Builder",
    "min_args": 1,
    "max_args": 1
}

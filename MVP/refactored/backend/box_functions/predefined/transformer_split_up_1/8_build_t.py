import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, embedding_layer, positional_encoding_layer, encoder_stack, decoder_stack, output_head):
        """
        embedding_layer: nn.Module mapping input tokens to embeddings.
        positional_encoding_layer: nn.Module that produces positional encodings.
        encoder_stack: list of encoder layers (each built by your EncoderLayerBuilder).
        decoder_stack: list of decoder layers (each built by your DecoderLayerBuilder).
        output_head: nn.Module mapping decoder outputs to vocabulary logits.
        """
        super(TransformerModel, self).__init__()
        self.embedding = embedding_layer
        self.pos_encoding = positional_encoding_layer

        # Wrap encoder and decoder stacks in ModuleList so that parameters are registered.
        self.encoder_stack = nn.ModuleList(encoder_stack)
        self.decoder_stack = nn.ModuleList(decoder_stack)
        self.output_head = output_head

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Embed the source and target tokens and add positional encodings.
        src_emb = self.pos_encoding(self.embedding(src))
        tgt_emb = self.pos_encoding(self.embedding(tgt))

        # Pass the source embeddings through the encoder stack.
        enc_output = src_emb
        for encoder_layer in self.encoder_stack:
            enc_output = encoder_layer(enc_output, mask=src_mask, key_padding_mask=src_key_padding_mask)

        # In a complete Transformer, the decoder would use cross-attention to attend to enc_output.
        # Since our custom decoder layer as built here does not include cross-attention,
        # we only pass the target embeddings through the decoder stack.
        dec_output = tgt_emb
        for decoder_layer in self.decoder_stack:
            dec_output = decoder_layer(dec_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # Compute final outputs (e.g., logits over the vocabulary).
        output = self.output_head(dec_output)
        return output


def invoke(builder):
    model = builder.build_model()
    return model


meta = {
    "name": "8 Build T",
    "min_args": 1,
    "max_args": 1
}

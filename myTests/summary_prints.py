from torchinfo import summary
import torch
import math
import torch.nn as nn

# for transformer with 5 inputs (vocab_size, model_dim, num_heads, num_encoder_layers, num_decoder_layers)
if __name__ == "__main__":
    from torchinfo import summary
    vocab_size = 1000
    model_dim = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1

    model = main(vocab_size, model_dim, num_heads, num_encoder_layers, num_decoder_layers)

    src_seq_len = 10  # length of source sequence
    tgt_seq_len = 10  # length of target sequence
    batch_size = 32

    src = torch.randint(0, vocab_size, (src_seq_len, batch_size))
    tgt = torch.randint(0, vocab_size, (tgt_seq_len, batch_size))

    # summary
    summary(model, input_data=(src, tgt))


# for transformer with 5 inputs (model_dim, vocab_size, num_heads, num_encoder_layers, num_decoder_layers)
if __name__ == "__main__":
    vocab_size = 1000
    model_dim = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1

    model = main(model_dim, vocab_size, num_heads, num_encoder_layers, num_decoder_layers)

    # dummy input data.
    src_seq_len = 10  # length of source sequence
    tgt_seq_len = 10  # length of target sequence
    batch_size = 32

    src = torch.randint(0, vocab_size, (src_seq_len, batch_size))
    tgt = torch.randint(0, vocab_size, (tgt_seq_len, batch_size))

    # summary
    summary(model, input_data=(src, tgt))


# for transformer with 2 inputs (model_dim, vocab_size)
if __name__ == "__main__":
    vocab_size = 1000
    model_dim = 512

    model = main(model_dim, vocab_size)

    src_seq_len = 10  # length of source sequence
    tgt_seq_len = 10  # length of target sequence
    batch_size = 32

    src = torch.randint(0, vocab_size, (src_seq_len, batch_size))
    tgt = torch.randint(0, vocab_size, (tgt_seq_len, batch_size))

    # summary
    summary(model, input_data=(src, tgt))


# for normal
if __name__ == "__main__":
    model = main()
    model.summary()


if __name__ == "__main__":
    from torchinfo import summary
    vocab_size = 1000
    model_dim = 512

    model = main(model_dim, vocab_size)

    src_seq_len = 10  # length of source sequence
    tgt_seq_len = 10  # length of target sequence
    batch_size = 32

    src = torch.randint(0, vocab_size, (src_seq_len, batch_size))
    tgt = torch.randint(0, vocab_size, (tgt_seq_len, batch_size))

    summary(model, input_data=(src, tgt))

# for FFN
if __name__ == "__main__":
    model = main(10)
    summary(
        model,
        input_size=(8, 10),
        col_names=("input_size", "output_size", "num_params")
    )

# for CNN
if __name__ == "__main__":
    model = main((3, 32, 32))
    summary(
        model,
        input_size=(8, 3, 32, 32),
        col_names=("input_size", "output_size", "num_params")
    )

# for RNN
if __name__ == "__main__":
    model = main(10)
    summary(
        model,
        input_size=(8, 20, 10),
        col_names=("input_size", "output_size", "num_params")
    )

# Hyperparameters and Meanings

## 1. vocab_size: 
    The size of vocabulary, the total number of distinct tokens (words, subwords, characters) that the model will handle.
    
    Usage:
    
    Embedding Layer: 
        vocab_size determines the number of rows in the embedding matrix (one roe per token).
    
    Final Linear Layer: 
        vocab_size projects the model's output into a vector of size vocab_size 
        representing logits (or probabilities after softmax) for each possible token.

## 2. model_dim:
    The dimentionality of the hidden representations (also called the embedding dimension).
    
    Usage:
    
    Embedding Layer:
        Each token is mapped to a vector of size model_dim.
    
    Transformer Module: 
        model_dim defines the dimention of the inputs/outputs of the transformer's internal layers.
    
    Scaling Factor:
        In the forward pass, the embeddings are scaled by √model_dim to help stabilize training.

## 3. num_heads:
    The number of attention heads in the multi-head self-attention mechanism.

    Usage:
        In the transformer module the attention mechanism is split into multiple "heads". 
        Each head learns different aspects of the input sequence and their outputs are later concatenated.

## 4. num_encoding_layers:
    The number of layers (or blocks) in the encoder part of the transformer.

    Usage:
        Each encoder layer contains self-attention and feed-forward sublayers. 
        More layers typically allow the model to capture more complex patterns.

## 5. num_decoder_layers:
    The number of layers in the decoder part of the transformer.
    
    Usage:
        The decoder layers not only perform self-attention but also include cross-attention with the encoder’s output, 
        making them crucial for tasks like sequence-to-sequence generation.

## 6. dropout
    The dropout probability used in various parts of the model (e.g., inside the Positional Encoding and the Transformer module).

    Usage:
        Dropout helps prevent overfitting by randomly setting a fraction of the activations to zero during training.

## 7. max_len
    The maximum sequence length for which positional encodings are precomputed.

    Usage:
        Positional Encoding Module:
            A matrix of shape (max_len, model_dim) is created to store the positional encodings. 
            Each row represents the encoding for a specific position in the input sequence.

        Considerations:
            Ensure that max_len is set to a value high enough to cover the longest sequences expected during training or inference. 
            If a sequence exceeds max_len, the module will only have encodings for the first max_len positions.

# Components of the Model

## 1. Embedding Layer (self.embedding):
    Converts token indices (integers) into dense vectors of size model_dim.

## 2. PositionalEncoding (self.pos_encoder):
    Adds information about the position of each token in the sequence.

    How it works:
        A pre-computed matrix of sinusoidal values is added to the embeddings, 
        allowing the model to differentiate between tokens based solely on their order.

## 3. Transformer Module (self.transformer):
    The core component that processes the input sequences using self-attention and feed-forward networks.
    
    Functionality:
        It consists of multiple encoder and decoder layers, 
        each learning different representations and relationships among tokens.

## 4. Final Linear Layer (self.fc_out):
    Projects the output of the transformer (which is still of dimension model_dim) to a vector of size vocab_size.
    
    Usage:
        This is used to compute scores (logits) for each token in the vocabulary.

## 5. LogSoftmax (self.softmax):
    Applies the log-softmax function to the final output.
    
    Usage:
        It converts the logits into log-probabilities, which are useful when training with a loss like the negative log-likelihood.

# Forward Pass Variables

## 1. src (source sequence):
    A tensor of shape (src_seq_len, batch_size) containing the indices of the tokens in the source sequence.

    Example:
        If you’re translating from English to French, src might represent the English sentence as a sequence of token indices.

## 2. tgt (target sequence):
    A tensor of shape (tgt_seq_len, batch_size) containing the indices of the tokens in the target sequence.

    Example:
        In a translation task, tgt would be the French sentence.

## 3. src_mask:
    An optional mask applied to the source sequence.

    Purpose:
        Masks can be used to ignore padding tokens or to control which tokens the model should attend to.

## 4. tgt_mask:
    A mask for the target sequence, typically used in autoregressive generation.

    Purpose:
        It ensures that, at each step, the model only attends to previously generated tokens (or the current token) and not future ones.

# How It All Fits Together

## 1. Embedding and Scaling:
    Both src and tgt pass through the embedding layer, converting indices into dense vectors.
    The embeddings are scaled by multiplying with √model_dim to maintain appropriate variance.

## 2. Adding Positional Information:
    The scaled embeddings are then passed through the PositionalEncoding module, which adds the sinusoidal positional information.

## 3. Transformer Processing:
    The processed embeddings for both source and target are fed into the transformer module, 
    where multiple layers of attention and feed-forward operations take place.

## 4. Projection and Output:
    The output of the transformer is passed through the linear layer to map the hidden state back to the vocabulary size.
    Finally, LogSoftmax is applied to produce log-probabilities over the vocabulary for each token in the target sequence.

## 5. Inference/Training:
    The model’s forward method takes in the source and target sequences (along with optional masks) and returns the predicted log-probabilities, 
    which can then be used for training or inference.

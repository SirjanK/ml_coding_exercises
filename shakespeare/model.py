import torch

"""
In the code comments, we refer:
        - B := batch size
        - V := vocabulary size
        - T := block size (context length)
        - E := embedding size
        - H := number of heads
"""


class CausalMultiHeadSelfAttention(torch.nn.Module):
    """
    Implementation of the masked multi-head self-attention mechanism module.
    """

    def __init__(self, embedding_size: int, num_heads: int, block_size: int):
        """
        Initialize the MaskedMultiHeadSelfAttention module.

        :param embedding_size: The size of the input embeddings.
        :param num_heads: Number of attention heads.
        :param block_size: The maximum context length for the model.
        """
        super(CausalMultiHeadSelfAttention, self).__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.block_size = block_size 

        assert self.embedding_size % self.num_heads == 0, "Embedding size must be divisible by number of heads."
        self.head_size = self.embedding_size // self.num_heads

        self._construct_parameters()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        :param input: Input tensor of shape (B, T, E)
        :return: Output tensor of shape (B, T, E)
        """

        # input shape: B x T x E
        _, T, _ = input.shape

        # rotate input for RoPE
        rope_rotated_input = self.rotate_for_position(input, self.sin[:T, :], self.cos[:T, :])  # B x T x E

        # compute Q, K, V projections - Q, K applied to rope, V to input
        q = self.q_proj(rope_rotated_input)  # B x T x E
        k = self.k_proj(rope_rotated_input)  # B x T x E
        v = self.v_proj(input)  # B x T x E

        return self.mhsa_with_qkv(q, k, v, self.mask[:, :, :T, :T])  # B x T x E
    
    def _construct_parameters(self):
        """
        Construct the parameters for the masked self-attention module.
        """ 
        # cached RoPE tables for positional encodings
        thetas = 10000 ** (-torch.arange(0, self.embedding_size, 2, dtype=torch.float32) / self.embedding_size)  # (E/2,)
        # repeat thetas
        thetas = torch.repeat_interleave(thetas, 2)  # (E,)
        self.register_buffer("thetas", thetas)
        # precompute sin and cosines for RoPE
        expanded_thetas = thetas.unsqueeze(0).repeat(self.block_size, 1)  # (T, E)
        scaled_thetas = torch.arange(self.block_size, dtype=torch.float32).unsqueeze(1) * expanded_thetas  # (T, E)
        sin = torch.sin(scaled_thetas)  # (T, E)
        cos = torch.cos(scaled_thetas)  # (T, E)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

        # Q, K, V projection matrices for all heads (wrapped into a single projection matrix for efficiency) (E, E) for each
        self.q_proj = torch.nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.k_proj = torch.nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.v_proj = torch.nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        # mask for the dot products before attention
        mask = torch.tril(torch.ones(self.block_size, self.block_size))  # T x T
        mask = mask.view(1, 1, self.block_size, self.block_size)  # 1 x 1 x T x T
        self.register_buffer("mask", mask)

        # dropout for attention matrix
        self.attention_dropout = torch.nn.Dropout(0.2)

        # output projection matrix (concat head outputs and pass through this)
        self.output_proj = torch.nn.Linear(self.embedding_size, self.embedding_size, bias=True)  # E x E

        # dropout for output projection
        self.output_dropout = torch.nn.Dropout(0.2)
    
    def mhsa_with_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dot_product_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked multi-head self-attention output using pre-computed Q, K, V.

        :param q: Query tensor of shape (B, T_q, E)
        :param k: Key tensor of shape (B, T_kv, E)
        :param v: Value tensor of shape (B, T_kv, E)
        :param dot_product_mask: Mask tensor of shape (1, 1, T_q, T_kv) to apply to the dot product. 0s indicate where we don't want attention
        to be computed (i.e. filled with -inf before softmax).
        :return: Output tensor of shape (B, T_q, E)
        """

        B, T_q, E = q.shape
        T_kv = k.shape[1]

        # reshape to separate out the heads
        q = q.view(B, T_q, self.num_heads, self.head_size).transpose(1, 2)  # B x H x T_q x E/H
        k = k.view(B, T_kv, self.num_heads, self.head_size).transpose(1, 2)  # B x H x T_kv x E/H
        v = v.view(B, T_kv, self.num_heads, self.head_size).transpose(1, 2)  # B x H x T_kv x E/H

        # compute dot product
        dot_product = q @ k.transpose(2, 3)  # B x H x T_q x T_kv
        # apply mask to the dot product
        masked_dot_product = dot_product.masked_fill(dot_product_mask == 0, float("-inf"))  # B x H x T_q x T_kv
        # scale by head size
        scaled_dot_product = masked_dot_product / (self.head_size ** 0.5)  # B x H x T_q x T_kv
    
        # apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)  # B x H x T_q x T_kv
        # apply dropout to attention weights
        attention_weights = self.attention_dropout(attention_weights)  # B x H x T_q x T_kv

        # compute the attention output
        attention_output = attention_weights @ v  # B x H x T_q x E/H

        # concat the head outputs
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T_q, E)  # B x T_q x E

        # apply the output projection and dropout
        output = self.output_proj(attention_output)  # B x T_q x E
        output = self.output_dropout(output)

        return output
    
    def rotate_for_position(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        Rotate the input tensor using the precomputed sine and cosine values for RoPE.

        :param x: Input tensor of shape (B, T, E).
        :param sin: Sine tensor of shape (T, E).
        :param cos: Cosine tensor of shape (T, E).
        :return: Rotated tensor of shape (B, T, E).
        """

        B, T, E = x.shape

        # chunk x appropriately to apply sine scaling
        reshaped_x = x.view(B, T, E // 2, 2)
        # flip the last dimension
        reshaped_x = reshaped_x[..., [1, 0]]
        # negate the first of each pair
        reshaped_x[..., 0] = -reshaped_x[..., 0]
        # flatten the last dimension
        reshaped_x = reshaped_x.view(B, T, E)

        # apply sine and cosine scaling
        return x * cos + reshaped_x * sin


class FeedForwardNetwork(torch.nn.Module):
    """
    Simple feed-forward network module applied to each token independently after each MhSA layer.
    """

    def __init__(self, embedding_size: int, scale_factor: int = 4):
        """
        Initialize the FeedForwardNetwork module.

        :param embedding_size: The size of the input embeddings.
        :param scale_factor: The scale factor for the hidden layer size.
        """
        super(FeedForwardNetwork, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = self.embedding_size * scale_factor

        self._construct_parameters()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        :param input: Input tensor of shape (B, T, E)
        :return: Output tensor of shape (B, T, E)
        """
        return self.mlp(input)
    
    def _construct_parameters(self):
        """
        Construct the parameters for the feed-forward network.
        """

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.hidden_size),  # E x 4E
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.embedding_size),  # 4E x E
            torch.nn.Dropout(0.2),
        )


class TransformerBlock(torch.nn.Module):
    """
    Transformer block consisting of a masked multi-head self-attention layer followed by a feed-forward network
    with appropriate residual connections and layer normalization.
    """

    def __init__(self, embedding_size: int, num_heads: int, block_size: int):
        """
        Initialize the TransformerBlock module.

        :param embedding_size: The size of the input embeddings.
        :param num_heads: Number of attention heads.
        :param block_size: The maximum context length for the model.
        """
        super(TransformerBlock, self).__init__()

        self.layer_norm1 = torch.nn.LayerNorm(embedding_size)
        self.mhsa = CausalMultiHeadSelfAttention(embedding_size, num_heads, block_size)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_size)
        self.ffn = FeedForwardNetwork(embedding_size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        :param input: Input tensor of shape (B, T, E)
        """

        # residual connections
        x = input + self.mhsa(self.layer_norm1(input))
        x = x + self.ffn(self.layer_norm2(x))

        return x


class ShakespeareGPT(torch.nn.Module):
    """
    A simple GPT model for generating text based on the works of Shakespeare.
    This model is a simplified version of the GPT architecture from the nano GPT repository. We reimplement it here for practice and may not exactly
    match the original implementation. 
    """

    def __init__(self, vocab_size: int, block_size: int, embedding_size: int, num_layers: int, num_heads: int):
        """
        Initialize the ShakespeareGPT model.

        :param vocab_size: The size of the vocabulary.
        :param block_size: The maximum context length for the model.
        :param embedding_size: The size of the embedding layer.
        :param num_layers: The number of transformer layers in the model.
        :param num_heads: The number of attention heads in the model.
        """
        super(ShakespeareGPT, self).__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self._construct_model()

        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model returning logits for the next token predictions.

        :param input: Input tensor of shape (B, T)
        :return: Logits tensor of shape (B, T, V) containing the predicted token logits.
        """

        # gather token embeddings
        token_embeddings = self.get_token_embeddings(input)  # B x T x E

        # apply transformer blocks
        for transformer_block in self.transformer_blocks:
            token_embeddings = transformer_block(token_embeddings)

        # pass through the MLP to get logits
        logits = self.token_prediction_proj(token_embeddings)  # B x T x V

        return logits

    def loss_fn(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the model.
        
        :param logits: Logits tensor of shape (B, T, V) containing the predicted token logits.
        :param target: Target tensor of shape (B, T) containing the true token indices.
        :return: Loss value.
        """

        # reshape logits and target to compute cross-entropy loss
        B, T, V = logits.shape
        # collapse batch and time dimensions
        logits = logits.view(B * T, V)
        target = target.view(B * T)

        return self.loss(logits, target)
    
    def _construct_model(self):
        """
        Construct the model architecture.
        """

        # embedding table for each token
        self.token_embedding_table = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
        ) 

        # gather transformer blocks
        transformer_blocks = []
        for _ in range(self.num_layers):
            transformer_blocks.append(
                TransformerBlock(
                    embedding_size=self.embedding_size,
                    num_heads=self.num_heads,
                    block_size=self.block_size,
                )
            )

        self.transformer_blocks = torch.nn.ModuleList(transformer_blocks)

        # Final projection for taking the final token embedding and producing the logits
        self.token_prediction_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.vocab_size),
        )
    
    def get_token_embeddings(self, input: torch.Tensor) -> torch.Tensor:
        """
        Get the token embeddings for the input tokens

        :param input: Input tensor of shape (B, T)
        :return: Token embeddings tensor of shape (B, T, E)
        """

        # gather token embeddings
        token_embeddings = self.token_embedding_table(input)  # B x T x E

        return token_embeddings 

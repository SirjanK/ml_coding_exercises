import torch


class ShakespeareGPT(torch.nn.Module):
    """
    A simple GPT model for generating text based on the works of Shakespeare.
    This model is a simplified version of the GPT architecture from the nano GPT repository. We reimplement it here for practice and may not exactly
    match the original implementation.

    In the code comments, we refer:
        - B := batch size
        - V := vocabulary size
        - T := block size (context length)
        - E := embedding size
    """

    def __init__(self, vocab_size: int, block_size: int, embedding_size: int):
        """
        Initialize the ShakespeareGPT model.

        :param vocab_size: The size of the vocabulary.
        :param block_size: The maximum context length for the model.
        :param embedding_size: The size of the embedding layer.
        """
        super(ShakespeareGPT, self).__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_size = embedding_size

        self._construct_model()

        self.loss = torch.nn.CrossEntropyLoss()

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model returning logits for the next token predictions.

        :param input: Input tensor of shape (B, T)
        :return: Logits tensor of shape (B, T, V) containing the predicted token logits.
        """

        # gather token embeddings
        token_embeddings = self.token_embedding_table(input)  # B x T x E

        # average the token embeddings along the time dimension
        averaged_tokens = self.averaging_mask @ token_embeddings  # B x T x E

        # pass through the MLP to get logits
        logits = self.mlp(averaged_tokens)  # B x T x V

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

        # causal averaging mask
        triangular_mask = torch.tril(torch.ones((self.block_size, self.block_size)))
        averaging_mask = triangular_mask / triangular_mask.sum(dim=-1, keepdim=True)
        self.register_buffer("averaging_mask", averaging_mask)  # T x T

        # MLP for taking the averaged token embeddings and predicting the next token logit
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.vocab_size),
        )

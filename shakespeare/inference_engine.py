import torch
from shakespeare.model import ShakespeareGPT, CausalMultiHeadSelfAttention, TransformerBlock
from typing import Optional


class KVCache:
    """
    A simple KV cache for transformer blocks in the ShakespeareGPT model.

    We store the key and value tensors along with the length of these. We define a max length that is the limit for the size of this cache.
    """

    def __init__(self, max_length: int, embedding_size: int, init_k: Optional[torch.Tensor] = None, init_v: Optional[torch.Tensor] = None):
        """
        Initialize the KV cache.

        :param max_length: The maximum length of the KV cache
        :param init_k: Optional initial key tensor of shape (1, K, E)
        :param init_v: Optional initial value tensor of shape (1, K, E)
        """
        assert max_length > 0, "max_length must be greater than 0"
        assert not ((init_k is None) ^ (init_v is None)), "Either both init_k and init_v should be None or both should be provided"
        if init_k is not None:
            assert init_k.shape == init_v.shape, "init_k and init_v must have the same shape"
            assert init_k.shape[0] == 1, "init_k and init_v must have batch size 1"
            assert init_k.shape[2] == embedding_size, "init_k and init_v must have embedding size E"
            assert init_k.shape[1] <= max_length, "init_k and init_v must have length less than or equal to max_length"
        self.max_length = max_length
        # dictionary from transformer block to a tuple of the cached key and value tensors (initially empty)
        self.k = init_k if init_k is not None else torch.empty((1, 0, embedding_size), dtype=torch.float32)
        self.v = init_v if init_v is not None else torch.empty((1, 0, embedding_size), dtype=torch.float32)
    
    def __len__(self) -> int:
        """
        Get the current length of the KV cache.

        :return: The current length of the KV cache
        """
        return self.k.shape[1]
    
    def add(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Add a new key and value tensor to the KV cache.

        :param k: The key tensor to add of shape (1, 1, E)
        :param v: The value tensor to add of shape (1, 1, E)
        """
        # check if the cache is full
        if len(self) >= self.max_length:
            # prune the first token
            self.k = self.k[:, -self.max_length + 1:, :]
            self.v = self.v[:, -self.max_length + 1:, :]

        # concat K, V to the cache
        self.k = torch.cat((self.k, k), dim=1)
        self.v = torch.cat((self.v, v), dim=1)


class InferenceEngine:
    """
    This custom inference engine supports efficient rolling next token inference for the ShakespeareGPT model
    via KV caching and minimal inference compute.

    It is specialized to support the use case of recurringly calling ShakespeareGPT inference to generate next token
    that we then add to the context for the next inference call.
    """

    def __init__(self, model: ShakespeareGPT, context: Optional[torch.Tensor] = None):
        """
        Initialize the inference engine

        :param model: The trained ShakespeareGPT model to use for inference
        :param context: Initial context for the model - usually the prompt[:-1] with the last token reserved for the inference call
        """

        self.model = model
        self.model.eval()

        self.block_size = model.block_size

        if context is not None and context.shape[1] > self.block_size - 1:
            # prune to get T-1 tokens
            context = context[:, -self.block_size + 1:]
        
        if context is None:
            self.pos = 0  # next token will be in position 0
            # initialize the KV cache to empty tensors of desired shape
            self.kv_caches = {
                transformer_block.mhsa: KVCache(max_length=self.block_size, embedding_size=model.embedding_size)
                for transformer_block in model.transformer_blocks
            }
        else:
            self.pos = context.shape[1]  # next token will be in this starting position
            x = self.model.get_token_embeddings(context)
            # dictionary from transformer block to the cached key and value tensors for the current context
            self.kv_caches = dict()
            self._construct_init_kv_cache(x)    

    @torch.no_grad()
    def inference(self, next_token: int) -> torch.Tensor:
        """
        Run inference on the context appended with next_token. Prune context if needed.

        :param next_token: The next token to append to the context for inference
        :return: The model's output logits for the next token, shape (1, V)
        """

        x = self._compute_token_embedding(next_token)  # 1 x 1 x E

        for transformer_block in self.model.transformer_blocks:
            # run inference on single transformer block using kv cache
            x = self._inference_transformer_block(x, transformer_block)  # 1 x 1 x E

        # pass through final projection layer
        x = torch.squeeze(self.model.token_prediction_proj(x), dim=0)  # 1 x V

        self.pos += 1

        return x
    
    def _construct_init_kv_cache(self, x: torch.Tensor) -> None:
        """
        Construct initial KV cache for the transformer blocks using the initial context.

        :param x: token embeddings for initial context shape (1, T, E)
        """

        _, T, E = x.shape

        # we undergo a forward pass through the transformer blocks, extracting the KV cache each time
        for transformer_block in self.model.transformer_blocks:
            x_tmp = transformer_block.layer_norm1(x)  # apply layer norm
            k = transformer_block.mhsa.k_proj(x_tmp)  # 1 x T x E
            v = transformer_block.mhsa.v_proj(x_tmp)  # 1 x T x E
            # rotate k
            k = transformer_block.mhsa.rotate_for_position(
                k, 
                sin=transformer_block.mhsa.sin[:T, :],
                cos=transformer_block.mhsa.cos[:T, :],
            )  # 1 x T x E
            # concat K, V to the cache
            self.kv_caches[transformer_block.mhsa] = KVCache(max_length=self.block_size, embedding_size=E, init_k=k, init_v=v)
            # apply the full transformer block to x
            x = transformer_block(x)
    
    def _compute_token_embedding(self, token: int) -> torch.Tensor:
        """
        Compute the token embedding for a given token.

        :param token: The token to compute the embedding for
        :return: The token embedding, shape 1 x 1 x E
        """

        # get token embedding
        x = torch.tensor([token], dtype=torch.long).unsqueeze(0)  # 1 x 1
        x = self.model.get_token_embeddings(x)  # 1 x 1 x E

        return x

    def _inference_transformer_block(self, x: torch.Tensor, transformer_block: TransformerBlock) -> torch.Tensor:
        """
        Inference on a single transformer block using the KV cache.

        :param x: The input tensor to the transformer block, shape 1 x 1 x E
        :param transformer_block: The transformer block to run inference on
        :return: The output tensor from the transformer block, shape 1 x 1 x E; the KV cache is updated internally
        """

        x = x + self._inference_mhsa(transformer_block.layer_norm1(x), transformer_block.mhsa)  # 1 x 1 x E
        x = x + transformer_block.ffn(transformer_block.layer_norm2(x))  # 1 x 1 x E

        return x

    def _inference_mhsa(self, x: torch.Tensor, mhsa: CausalMultiHeadSelfAttention) -> torch.Tensor:
        """
        Inference on a single MHSA layer using the KV cache.

        :param x: The input tensor to the MHSA layer, shape 1 x 1 x E
        :param mhsa: The MHSA layer to run inference on
        :return: The output tensor from the MHSA layer, shape 1 x 1 x E; the KV cache is updated internally
        """
 
        # compute q, k, v, (1, 1, E) for each
        curr_q = mhsa.q_proj(x)
        curr_k = mhsa.k_proj(x)
        curr_v = mhsa.v_proj(x)

        # rotate q, k
        scaled_thetas = mhsa.thetas * self.pos
        # (1, E) for each
        sin = torch.sin(scaled_thetas).unsqueeze(0)
        cos = torch.cos(scaled_thetas).unsqueeze(0)

        curr_q = mhsa.rotate_for_position(
            curr_q,
            sin=sin,
            cos=cos,
        )
        curr_k = mhsa.rotate_for_position(
            curr_k,
            sin=sin,
            cos=cos,
        )

        # concat k, v to the cache
        cache = self.kv_caches[mhsa]
        cache.add(curr_k, curr_v)

        # run inference
        T_kv = len(cache)
        mask = torch.ones(1, 1, 1, T_kv, dtype=torch.bool)  # attend to every position (T_q = 1)
        x = mhsa.mhsa_with_qkv(
            q=curr_q,
            k=cache.k,
            v=cache.v,
            dot_product_mask=mask,
        )

        return x

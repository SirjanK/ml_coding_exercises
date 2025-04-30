import torch
from shakespeare.model import ShakespeareGPT, CausalMultiHeadSelfAttention, TransformerBlock
from typing import Optional


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
            self.kv_cache = {
                transformer_block.mhsa: torch.empty((1, 0, 2 * model.embedding_size), dtype=torch.float32)
                for transformer_block in model.transformer_blocks
            }
        else:
            self.pos = context.shape[1]  # next token will be in this starting position
            x = self.model.get_token_embeddings(context)
            # dictionary from transformer block to the cached key and value tensors for the current context
            self.kv_cache = dict()
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

        self.pos = min(self.pos + 1, self.block_size - 1)  # increment position (max is block_size - 1)

        return x
    
    def _construct_init_kv_cache(self, x: torch.Tensor) -> None:
        """
        Construct initial KV cache for the transformer blocks using the initial context.

        :param x: token embeddings for initial context shape (1, T, E)
        """

        # we undergo a forward pass through the transformer blocks, extracting the KV cache each time
        for transformer_block in self.model.transformer_blocks:
            x_tmp = transformer_block.layer_norm1(x)  # apply layer norm
            qkv = transformer_block.mhsa.qkv_proj(x_tmp)  # 1 x T x 3E
            extracted_kv = qkv[:, :, self.model.embedding_size:]  # 1 x T x 2E
            self.kv_cache[transformer_block.mhsa] = extracted_kv  # store extracted KV in cache
            # apply the full transformer block to x
            x = transformer_block(x)
    
    def _compute_token_embedding(self, token: int) -> torch.Tensor:
        """
        Compute the token embedding for a given token.

        :param token: The token to compute the embedding for
        :param pos: position for the token
        :return: The token embedding, shape 1 x 1 x E
        """

        # get pos as a tensor
        position = torch.tensor([self.pos], dtype=torch.long).unsqueeze(0)  # 1 x 1

        # get token embedding
        x = torch.tensor(token, dtype=torch.long).unsqueeze(0)  # 1 x 1
        x = self.model.token_embedding_table(x)  # 1 x 1 x E
        x = x + self.model.positional_embedding_table(position)  # 1 x 1 x E

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

        E = x.shape[-1]

        # get QKV for the current input
        curr_qkv = mhsa.qkv_proj(x)  # 1 x 1 x 3E

        # split into Q, KV
        curr_q, curr_kv = curr_qkv[:, :, :E], curr_qkv[:, :, E:]  # 1 x 1 x E for q, 1 x 1 x 2E for kv 

        # concat k, v to the cache
        self.kv_cache[mhsa] = torch.cat((self.kv_cache[mhsa], curr_kv), dim=1)

        # split out K, V
        k, v = self.kv_cache[mhsa][:, :, :E], self.kv_cache[mhsa][:, :, E:]

        # run inference
        T_kv = self.kv_cache[mhsa].shape[1]
        mask = torch.ones(1, 1, 1, T_kv, dtype=torch.bool)  # attend to every position (T_q = 1)
        x = mhsa.mhsa_with_qkv(curr_q, k, v, dot_product_mask=mask)

        # update the cache (if it is full, prune the first token)
        if self.kv_cache[mhsa].shape[1] >= self.block_size:
            self.kv_cache[mhsa] = self.kv_cache[mhsa][:, -self.block_size + 1:, :]
         
        return x

import torch
import torch.nn.functional as F
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

        self.context = context
        if self.context is not None and self.context.shape[1] > self.block_size - 1:
            # prune to get T-1 tokens
            self.context = self.context[:, -self.block_size + 1:]
        
        if self.context is None:
            # if no context is provided, set it to an empty tensor of desired shape
            self.context = torch.empty((1, 0), dtype=torch.long)

            # also initialize the KV cache to empty tensors of desired shape
            self.kv_cache = {
                transformer_block.mhsa: torch.empty((1, 0, 2 * model.embedding_size), dtype=torch.float32)
                for transformer_block in model.transformer_blocks
            }
        else:
            # apply the model's embedding layer to the context along with positional encoding
            x = self.model.get_token_embeddings(self.context)
            # dictionary from transformer block to the cached key and value tensors for the current context
            self.kv_cache = {
                transformer_block.mhsa: self._compute_kv(x, transformer_block.mhsa)
                for transformer_block in model.transformer_blocks
            }
    
    @torch.no_grad()
    def inference(self, next_token: int) -> torch.Tensor:
        """
        Run inference on the context appended with next_token. Prune context if needed.

        :param next_token: The next token to append to the context for inference
        :return: The model's output logits for the next token, shape (1, V)
        """

        # get current context length as a tensor
        curr_context_length = self.context.shape[1]
        position = torch.tensor([curr_context_length], dtype=torch.long).unsqueeze(0)  # 1 x 1

        # get token embedding
        x = torch.tensor(next_token, dtype=torch.long).unsqueeze(0)  # 1 x 1
        x = self.model.token_embedding_table(x)  # 1 x 1 x E
        x = x + self.model.positional_embedding_table(position)  # 1 x 1 x E

        for transformer_block in self.model.transformer_blocks:
            # run inference on single transformer block using kv cache
            x = self._inference_transformer_block(x, transformer_block)  # 1 x 1 x E

        # pass through final projection layer
        x = torch.squeeze(self.model.token_prediction_proj(x), dim=0)  # 1 x V

        return x
    
    def _compute_kv(self, x: torch.Tensor, mhsa: CausalMultiHeadSelfAttention) -> torch.Tensor:
        """
        Compute the key and value tensors for the multi-head self-attention layer on the current context.

        :param x: The input tensor to the multi-head self-attention layer, shape B x T x E
        :param mhsa: The multi-head self-attention layer
        :return: tensor of shape B x T' x 2E where T' is the current length of the context (guaranteed to be <= T - 1). The first of E dims
        refers to the keys, second to the values
        """

        E = x.shape[-1]

        # get the linear projection matrix for K, V - we have no bias here
        weight_kv = mhsa.qkv_proj.weight[E:, :]

        kv_proj = F.linear(x, weight_kv)  # B x T x 2E

        return kv_proj

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

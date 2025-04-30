import torch
import os
import pytest
from typing import Optional, Tuple
from shakespeare.inference_engine import InferenceEngine
from shakespeare.model import ShakespeareGPT
from shakespeare.train import load_vocab, get_config, MODEL_PATH
from shakespeare.tokenizer import Tokenizer


# test the inference engine by sampling 300 tokens using full inference and the inference engine
# fix the manual seed for fair comparison


def setup_inference_engine(prompt: Optional[str] = None) -> Tuple[torch.Tensor, InferenceEngine]:
    """
    Setup the inference engine with the given prompt.

    :return: Tuple of the encoded context and the inference engine
    """

    config = get_config('lite')
    vocab = load_vocab("shakespeare/data")
    tokenizer = Tokenizer(vocab)
    model = ShakespeareGPT(
        vocab_size=len(vocab),
        block_size=config.block_size,
        embedding_size=config.embedding_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "lite/best_model.pth")))
    model.eval()

    # initial context
    expanded_prompt = prompt if prompt is not None else " "
    context = torch.tensor(tokenizer.encode(expanded_prompt), dtype=torch.long).unsqueeze(0)  # batch size 1

    # create the inference engine
    if prompt is None:
        inference_engine = InferenceEngine(model, context=None)
    else:
        inference_engine = InferenceEngine(model, context=context[:, :-1])  # reserve the last token for inference
    
    return context, inference_engine


def test_compute_token_embedding():
    """
    Test the _compute_token_embedding function in the InferenceEngine class.
    """
    context, engine = setup_inference_engine("Long live the")
    model = engine.model

    # get the token embedding normally for the context plus a next token
    expected_token_embeddings = model.get_token_embeddings(context)[:, -1, :]

    # run the _compute_token_embedding function
    next_token = context[0, -1].item()
    token_embedding = engine._compute_token_embedding(next_token)[:, 0, :]

    # assert equivalence
    assert torch.allclose(expected_token_embeddings, token_embedding, atol=1e-5), f"Token embedding mismatch: {expected_token_embeddings} vs {token_embedding}"


def test_inference_mhsa():
    """
    Test the _inference_mhsa function in the InferenceEngine class.
    """
    context, engine = setup_inference_engine("Long live the")
    model = engine.model

    # get the first mhsa
    transformer_block = model.transformer_blocks[0]
    mhsa = transformer_block.mhsa

    # get token embedding normally and pass through MHSA
    token_embeddings = model.get_token_embeddings(context)  # (1, T, E)
    expected_output = mhsa(transformer_block.layer_norm1(token_embeddings))[:, -1, :]  # (1, E) using the last token's output

    # run the inference engine
    next_token = context[0, -1].item()
    curr_token_embedding = engine._compute_token_embedding(next_token)  # (1, 1, E)
    inference_output = engine._inference_mhsa(transformer_block.layer_norm1(curr_token_embedding), mhsa)  # (1, 1, E)

    # assert equivalence
    assert torch.allclose(expected_output, inference_output, atol=1e-5), f"MHSA output mismatch: {expected_output} vs {inference_output}"


def test_inference_transformer_blocks():
    """
    Test the _inference_transformer_block function in the InferenceEngine class.
    """
    context, engine = setup_inference_engine("Long live the")
    model = engine.model

    # get the first transformer block
    transformer_block = model.transformer_blocks[0]

    # get token embedding normally and pass through transformer block
    token_embeddings = model.get_token_embeddings(context)  # (1, T, E)
    expected_output = transformer_block(token_embeddings)

    # run the inference engine
    next_token = context[0, -1].item()
    curr_token_embedding = engine._compute_token_embedding(next_token)  # (1, 1, E)
    inference_output = engine._inference_transformer_block(curr_token_embedding, transformer_block)  # (1, 1, E)

    # assert equivalence
    assert torch.allclose(expected_output[:, -1, :], inference_output, atol=1e-5), f"Transformer block output mismatch: {expected_output[:, -1, :]} vs {inference_output}"

    # try for the next transformer block
    transformer_block = model.transformer_blocks[1]

    expected_output = transformer_block(expected_output)

    inference_output = engine._inference_transformer_block(inference_output, transformer_block)

    # assert equivalence
    assert torch.allclose(expected_output[:, -1, :], inference_output, atol=1e-5), f"Transformer #2 block output mismatch: {expected_output[:, -1, :]} vs {inference_output}"


# data provider for the prompt
@pytest.mark.parametrize("prompt", ["Long live the", None])
@torch.no_grad()
def test_inference_engine(prompt: Optional[str]):
    """
    Test end-to-end inference engine logic compared to naive inference.
    """
    torch.manual_seed(12)

    context, engine = setup_inference_engine(prompt=prompt)
    model = engine.model

    LENGTH = 2
    for _ in range(LENGTH):
        # trim context if it exceeds the model's block size
        if context.shape[1] > model.block_size:
            context = context[:, -model.block_size:]

        # run full model inference manually to get logits
        logits = model(context)  # (1, T', V) where T' is the length of the current context (T' <= T)

        # index into next_token_idx
        logits = logits[0, -1, :]  # (V,) using the last token's logit

        # run inference engine on the latest token logits
        inference_engine_logits = engine.inference(context[0, -1].item())

        # assert equivalence of logits
        assert torch.allclose(logits, inference_engine_logits, atol=1e-5), f"Logits mismatch: {logits} vs {inference_engine_logits}"

        # get the max logit index - that will be the one we use for the next token
        next_token = torch.argmax(inference_engine_logits).item()  # use inference engine logits (should be same given above assertion)
 
        # update the context and next token idx
        context = torch.cat((context, torch.tensor([[next_token]], dtype=torch.long)), dim=1) 

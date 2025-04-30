import torch
import os
from shakespeare.inference_engine import InferenceEngine
from shakespeare.model import ShakespeareGPT
from shakespeare.train import load_vocab, get_config, MODEL_PATH
from shakespeare.tokenizer import Tokenizer


# test the inference engine by sampling 10 tokens using full inference and the inference engine
# fix the manual seed for fair comparison


def test_inference_engine():
    torch.manual_seed(12)

    # load the lite model and tokenizer
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
    context = torch.tensor(tokenizer.encode("Long live the"), dtype=torch.long).unsqueeze(0)  # batch size 1

    # create the inference engine
    inference_engine = InferenceEngine(model, context[:, :-1])

    for _ in range(10):
        # trim context if it exceeds the model's block size
        if context.shape[1] > model.block_size:
            context = context[:, -model.block_size:]

        # run full model inference manually to get logits
        logits = model(context)  # (1, T', V) where T' is the length of the current context (T' <= T)

        # index into next_token_idx
        logits = logits[0, -1, :]  # (V,) using the last token's logit

        # run inference engine on the latest token logits
        inference_engine_logits = inference_engine.inference(context[0, -1].item())

        # assert equivalence of logits
        assert torch.allclose(logits, inference_engine_logits, atol=1e-5), f"Logits mismatch: {logits} vs {inference_engine_logits}"

        # get the max logit index - that will be the one we use for the next token
        next_token = torch.argmax(inference_engine_logits).item()  # use inference engine logits (should be same given above assertion)
 
        # update the context and next token idx
        context = torch.cat((context, torch.tensor([[next_token]], dtype=torch.long)), dim=1) 

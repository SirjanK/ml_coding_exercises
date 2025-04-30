import time
import torch
from argparse import ArgumentParser
from typing import Iterator, Optional

from shakespeare.model import ShakespeareGPT
from shakespeare.train import load_vocab, get_config
from shakespeare.tokenizer import Tokenizer


@torch.no_grad()
def text_generator(tokenizer: Tokenizer, model: ShakespeareGPT, prompt: str, length: int, topk_sampling_limit: Optional[int], temperature: float) -> Iterator[str]:
    """
    Generator for a character for the Shakespeare GPT model; returns a built text.

    :param tokenizer: Tokenizer object for encoding and decoding.
    :param model: ShakespeareGPT model object.
    :param prompt: Initial text prompt to start the generation.
    :param length: Length of the text to generate (in characters).
    :param topk_sampling_limit: Optional top-k sampling limit for the generation. if not specified, we do random sampling from full distribution
    :param temperature: temperature for sampling. Pass 1.0 if you don't want to scale the logits.
    """

    # list of encoded tokens so far
    # we start off with the prompt encoded (tokenizer.encode() will error if the prompt is not in the vocab)
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)  # batch size 1

    for _ in range(length):
        # trim context if it exceeds the model's block size
        if context.shape[1] > model.block_size:
            context = context[:, -model.block_size:]

        # run model inference to get logits
        logits = model(context)  # (1, T', V) where T' is the length of the current context (T' <= T)

        # index into next_token_idx
        logits = logits[0, -1, :]  # (V,) using the last token's logit

        # apply temperature scaling
        logits = logits / temperature

        # sample from the logits (multinomial distribution)
        probs = torch.softmax(logits, dim=-1)  # (V,)
        # sort probs and get top-k indices
        if topk_sampling_limit is not None:
            topk_probs, topk_indices = torch.topk(probs, topk_sampling_limit)
            # sample from the top-k indices
            probs = torch.zeros_like(probs)
            probs[topk_indices] = topk_probs
            # normalize the probabilities
            probs /= probs.sum()
        next_token = torch.multinomial(probs, num_samples=1).item()

        # yield the decoded character
        yield tokenizer.decode_single(next_token)

        # update the context and next token idx
        context = torch.cat((context, torch.tensor([[next_token]], dtype=torch.long)), dim=1) 


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate text using the trained Shakespeare GPT model.")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory containing the vocab file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    ) 
    parser.add_argument(
        "--config",
        type=str,
        choices=["lite", "heavy"],
        required=True,
        help="Configuration to use for training the model.",
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Length of the text to generate (in characters).",
    )
    parser.add_argument(
        "--topk-sampling-limit",
        type=int,
        required=False,
        help="Top-k sampling limit for the generation (optional).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="Temperature for sampling (optional).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default=" ",  # prompt with simply a space
        help="Text prompt to start the generation (optional).",
    )

    args = parser.parse_args()

    # load the tokenizer
    vocab = load_vocab(args.data_path)
    tokenizer = Tokenizer(vocab)

    # load the model
    config = get_config(args.config)
    model = ShakespeareGPT(
        vocab_size=len(vocab),
        block_size=config.block_size,
        embedding_size=config.embedding_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()

    # while we can still generate
    print(args.prompt, end="")
    # benchmarking for token generation
    start_time = time.time()
    for text in text_generator(tokenizer, model, args.prompt, args.length, args.topk_sampling_limit, args.temperature):
        print(text, end="")
    end_time = time.time()
    print(f"\n\nTime taken for generation: {(end_time - start_time) / args.length * 1000:.2f} milliseconds/token")

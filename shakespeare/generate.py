from argparse import ArgumentParser
from typing import Iterator
import torch

from shakespeare.model import ShakespeareGPT
from shakespeare.train import load_vocab, get_config
from shakespeare.tokenizer import Tokenizer


@torch.no_grad()
def text_generator(tokenizer: Tokenizer, model: ShakespeareGPT, length: int) -> Iterator[str]:
    """
    Generator for a character for the Shakespeare GPT model; returns a built text.
    """

    # list of encoded tokens so far
    # we start off with a space character only
    context = torch.tensor(tokenizer.encode(' '), dtype=torch.long).unsqueeze(0)  # batch size 1

    for _ in range(length):
        # run model inference to get logits
        logits = model(context)  # (1, T', V) where T' is the length of the current context (T' <= T)

        # index into next_token_idx
        logits = logits[0, -1, :]  # (V,) using the last token's logit

        # sample from the logits (multinomial distribution)
        probs = torch.softmax(logits, dim=-1)  # (V,)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # yield the decoded character
        yield tokenizer.decode_single(next_token)

        # update the context and next token idx
        # add the new token to the context
        context = torch.cat((context, torch.tensor([[next_token]], dtype=torch.long)), dim=1)
        if context.shape[1] > model.block_size:
            # trim
            context = context[:, -model.block_size:]


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
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # while we can still generate
    for text in text_generator(tokenizer, model, args.length):
        print(text, end="")

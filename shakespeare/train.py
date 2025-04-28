import os
import torch
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from shakespeare.configs import LITE_CONFIG, Config
from shakespeare.dataset import get_datasets
from shakespeare.model import ShakespeareGPT
from torchmetrics import Accuracy


ENCODED_DATA_FNAME = "encoded.npy"
VOCAB_FNAME = "vocab.txt"

LOGS_PATH = "logs"


def train(data_path: str, config: Config):
    """
    Train the Shakespeare GPT model using the provided configuration.
    """

    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = device.type == "cuda"

    # Initialize the dataloaders
    train_dataset, val_dataset = get_datasets(
        os.path.join(data_path, ENCODED_DATA_FNAME),
        block_size=config.block_size,
        validation_split=config.validation_split,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        pin_memory=is_gpu,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.validation_batch_size,
        pin_memory=is_gpu,
    )

    # Initialize the model
    vocab_size = len(open(os.path.join(data_path, VOCAB_FNAME)).read())
    model = ShakespeareGPT(
        vocab_size=vocab_size,
        block_size=config.block_size,
        embedding_size=config.embedding_size,
    )
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    # Setup tensorboard logging
    writer = SummaryWriter(
        os.path.join(LOGS_PATH, "shakespeare_gpt"),
        flush_secs=10,
    )

    # Training loop
    train_dataloader_iter = iter(train_dataloader)
    for iter in range(config.max_iters):
        # get batch of data
        input_data, target_data = next(train_dataloader_iter)
        input_data, target_data = input_data.to(device), target_data.to(device)

        # forward pass
        logits = model(input_data)
        loss = model.loss_fn(logits, target_data)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the loss (for training log every 100 steps)
        if iter % 100 == 0:
            writer.add_scalar("Loss/train", loss.item(), iter)
        if iter % config.eval_interval == 0:
            eval_loop(model, val_dataloader, iter, device, writer)


@torch.no_grad()
def eval_loop(
    model: ShakespeareGPT,
    val_dataloader: torch.utils.data.DataLoader,
    train_step: int,
    device: torch.device,
    writer: SummaryWriter,
) -> None:
    """
    Evaluate the model on the validation set and report the aggregated loss.

    :param model: The model to evaluate.
    :param val_dataloader: The dataloader for the validation set.
    :param train_step: Training step we are on.
    :param device: The device to run the evaluation on.
    :param writer: The tensorboard writer for logging.
    """
    model.eval()
    accuracy_metric = Accuracy(task="multiclass", top_k=5, num_classes=model.vocab_size)

    val_dataloader_iter = iter(val_dataloader)
    input_data, target_data = next(val_dataloader_iter)
    input_data, target_data = input_data.to(device), target_data.to(device)
    logits = model(input_data)
    B, T, V = logits.shape
    loss = model.loss_fn(logits, target_data)
    accuracy = accuracy_metric(logits.view(B * T, V), target_data.view(B * T))

    writer.add_scalar("Loss/validation", loss.item(), global_step=train_step)
    writer.add_scalar("Accuracy/validation", accuracy.item(), global_step=train_step)

    model.train()


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a Shakespeare GPT model.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data/ directory containing encoded dataset and vocabulary",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["lite", "heavy"],
        required=True,
        help="Configuration to use for training the model.",
    )

    args = parser.parse_args()

    if args.config == "lite":
        config = LITE_CONFIG
    else:
        raise ValueError("Unsupported configuration currently. Use 'lite' or 'heavy'.")
    
    # run training
    train(args.data_path, config)

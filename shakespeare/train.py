import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from shakespeare.configs import LITE_CONFIG, Config
from shakespeare.dataset import get_datasets
from shakespeare.model import ShakespeareGPT
from torchmetrics import Accuracy
from typing import List


ENCODED_DATA_FNAME = "encoded.npy"
VOCAB_FNAME = "vocab.txt"

LOGS_PATH = "shakespeare/logs"
MODEL_PATH = "shakespeare/model"


def load_vocab(data_path: str) -> List[str]:
    """
    Load the vocabulary from the specified path.
    """
    vocab_path = os.path.join(data_path, VOCAB_FNAME)
    with open(vocab_path, "r") as f:
        vocab = f.read()
    return list(vocab)


def get_config(config_name: str) -> Config:
    """
    Get the configuration for the specified model.
    """
    if config_name == "lite":
        return LITE_CONFIG
    else:
        raise ValueError("Unsupported configuration currently. Use 'lite' or 'heavy'.")


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
    vocab = load_vocab(data_path)
    vocab_size = len(vocab)
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
    # delete the logs/shakespeare_gpt directory if it exists
    dir_path = os.path.join(LOGS_PATH, "shakespeare_gpt")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    writer = SummaryWriter(
        dir_path,
        flush_secs=10,
    )

    # Training loop
    best_val_loss = float("inf")
    train_dataloader_iter = iter(train_dataloader)
    for i in range(config.max_iters):
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
        if i % 100 == 0:
            writer.add_scalar("Loss/train", loss.item(), i)
        if i % config.eval_interval == 0:
            loss = eval_loop(model, val_dataloader, i, device, writer)
            # store the model with best validation loss
            if loss < best_val_loss:
                best_val_loss = loss
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, "best_model.pth"))
    
    # save the final model
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "final_model.pth"))


@torch.no_grad()
def eval_loop(
    model: ShakespeareGPT,
    val_dataloader: torch.utils.data.DataLoader,
    train_step: int,
    device: torch.device,
    writer: SummaryWriter,
) -> float:
    """
    Evaluate the model on the validation set and report the aggregated loss.

    :param model: The model to evaluate.
    :param val_dataloader: The dataloader for the validation set.
    :param train_step: Training step we are on.
    :param device: The device to run the evaluation on.
    :param writer: The tensorboard writer for logging.
    :return: The loss value.
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
    writer.add_scalar("Top-5 Accuracy/validation", accuracy.item(), global_step=train_step)

    model.train()

    return loss.item()


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

    # run training
    train(args.data_path, get_config(args.config))

import numpy as np
import torch
from torch.utils.data import IterableDataset
from typing import Tuple


class ShakespeareDataset(IterableDataset):
    def __init__(self, data: np.ndarray, block_size: int):
        """
        The ShakespeareDataset class provides logic to randomly select a chunk of size block_size from the enocoded
        shakespeare dataset and provide them as torch tensors.

        :param data: numpy array containing the encoded dataset.
        :param block_size: Size of the blocks to be used for training.
        """

        # load torch tensor from encoded numpy file
        self.data = torch.from_numpy(data)
        assert self.data.dtype == torch.uint8, "Data type must be uint8"
        self.block_size = block_size

    def __iter__(self):
        while True:
            # get random start index
            start_index = np.random.randint(0, len(self.data) - self.block_size - 1)  # we read block_size + 1 bytes

            # get block_size + 1 chunk
            chunk = self.data[start_index : start_index + self.block_size + 1]

            # gather your input data along with target
            input_data = chunk[:-1]
            target_data = chunk[1:]

            yield input_data, target_data


def get_datasets(
    data_path: str,
    block_size: int,
    validation_split: float,
) -> Tuple[ShakespeareDataset, ShakespeareDataset]:
    """
    Get the datasets for training and validation.

    :param data_path: Path to the encoded dataset.
    :param block_size: Size of the chunks to be used for training.
    :param validation_split: Fraction of the dataset to be used for validation.
    :return: train and validation datasets.
    """

    data = np.load(data_path)

    VAL_SIZE = int(len(data) * validation_split)

    train_data = data[:-VAL_SIZE]
    val_data = data[-VAL_SIZE:]

    train_dataset = ShakespeareDataset(train_data, block_size)
    val_dataset = ShakespeareDataset(val_data, block_size)

    return train_dataset, val_dataset
    
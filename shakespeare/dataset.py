import numpy as np
import torch
from torch.utils.data import IterableDataset


class ShakespeareDataset(IterableDataset):
    def __init__(self, encoded_path: str, block_size: int):
        """
        The ShakespeareDataset class provides logic to randomly select a chunk of size block_size from the enocoded
        shakespeare dataset and provide them as torch tensors.

        :param encoded_path: Path to the encoded dataset in numpy format.
        :param block_size: Size of the blocks to be used for training.
        """

        # load torch tensor from encoded numpy file
        self.data = torch.from_numpy(np.load(encoded_path))
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

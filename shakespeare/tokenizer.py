import numpy as np
from typing import List


DATA_PATH = "shakespeare/data/input.txt"


def generate_vocabulary(text: str) -> List[str]:
    """
    Generate a vocabulary list from the given text.

    Args:
        text (str): The input text to generate vocabulary from.

    Returns:
        List[str]: A list of unique characters in the text.
    """

    return sorted(set(text))


class Tokenizer:
    def __init__(self, vocabulary: List[str]):
        """
        Initialize the Tokenizer with a vocabulary.

        Args:
            vocabulary (List[str]): The vocabulary list to use for tokenization.
        """
        self.vocab = vocabulary
        self.vocab_size = len(vocabulary)
        # since we encode using uint8_t, assert vocab size is less than 256
        assert self.vocab_size < 256, "Vocabulary size exceeds uint8_t limit (256)."
        self.char_to_id = {char: idx for idx, char in enumerate(vocabulary)}  # character to index map
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode the input text into a list of integers based on the vocabulary.

        Args:
            text (str): The input text to encode.

        Returns:
            np.ndarray: A uint8_t numpy array representing the encoded text.
        """
        
        encoded = []

        for char in text:
            if char not in self.char_to_id:
                raise ValueError(f"Character '{char}' not found in vocabulary.")
            encoded.append(self.char_to_id[char])
        
        return np.array(encoded, dtype=np.uint8)

    def decode(self, encoded_text: np.ndarray) -> str:
        """
        Decode a numpy array of integers back into the original text.

        Args:
            encoded_text (List[int]): The encoded text to decode.

        Returns:
            str: The decoded text.
        """

        decoded = []

        for idx in encoded_text:
            decoded.append(self.decode_single(idx))
        
        return ''.join(decoded)
    
    def decode_single(self, idx: int) -> str:
        """
        Decode a single index back into the original character.

        Args:
            idx (int): The index to decode.

        Returns:
            str: The decoded character.
        """

        if idx < 0 or idx >= self.vocab_size:
            raise ValueError(f"Index '{idx}' out of bounds for vocabulary size {self.vocab_size}.")
        
        return self.vocab[idx]


if __name__ == "__main__":
    # Read the input text
    with open(DATA_PATH, "r") as f:
        text = f.read()

    # Generate vocabulary
    vocab = generate_vocabulary(text)

    # Initialize the tokenizer
    tokenizer = Tokenizer(vocab)

    # Encode the text
    encoded_text = tokenizer.encode(text)

    # write out the vocabulary and encoded bytes
    with open("shakespeare/data/vocab.txt", "w") as f:
        for char in vocab:
            f.write(f"{char}")
    
    with open("shakespeare/data/encoded.npy", "wb") as f:
        # save into numpy format
        np.save(f, encoded_text)
        
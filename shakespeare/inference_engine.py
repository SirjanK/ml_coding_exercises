import torch
from model import ShakespeareGPT


class InferenceEngine:
    """
    This custom inference engine supports efficient rolling next token inference for the ShakespeareGPT model
    via KV caching and minimal inference compute.

    It is specialized to support the use case of recurringly calling ShakespeareGPT inference to generate next token
    that we then add to the context for the next inference call.
    """

    def __init__(self, model: ShakespeareGPT):
        """
        Initialize the inference engine

        :param model: The trained ShakespeareGPT model to use for inference
        """

        self.model = model
        self.model.eval()

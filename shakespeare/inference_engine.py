import torch
from shakespeare.model import ShakespeareGPT


class InferenceEngine:
    """
    This custom inference engine supports efficient rolling next token inference for the ShakespeareGPT model
    via KV caching and minimal inference compute.

    It is specialized to support the use case of recurringly calling ShakespeareGPT inference to generate next token
    that we then add to the context for the next inference call.
    """

    def __init__(self, model: ShakespeareGPT, context: torch.Tensor):
        """
        Initialize the inference engine

        :param model: The trained ShakespeareGPT model to use for inference
        :param context: Initial context for the model - usually the prompt[:-1] with the last token reserved for the inference call
        """

        self.model = model
        self.model.eval()

        self.block_size = model.block_size

        self.context = context
    
    @torch.no_grad()
    def inference(self, next_token: int) -> torch.Tensor:
        """
        Run inference on the context appended with next_token. Prune context if needed.

        :param next_token: The next token to append to the context for inference
        :return: The model's output logits for the next token
        """

        # prune context if needed
        if self.context.shape[1] > self.block_size:
            self.context = self.context[:, -self.block_size:]
        
        # append the next token to the context
        self.context = torch.cat((self.context, torch.tensor([[next_token]], dtype=torch.long)), dim=1)

        # run model inference to get logits
        logits = self.model(self.context)

        # index into latest token idx and return
        return logits[:, -1, :]  # (1, V): the last token's logit

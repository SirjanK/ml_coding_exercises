from dataclasses import dataclass


@dataclass
class Config:
    """Configuration base class for model construction and training."""
    batch_size: int
    block_size: int
    embedding_size: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    validation_split: float
    validation_batch_size: int


# Lite config for testing out model training - meant for training on CPU (e.g. on laptop)
LITE_CONFIG = Config(
    batch_size=32,
    block_size=8,
    embedding_size=64,
    learning_rate=1e-3,
    max_iters=100000,
    eval_interval=1000,
    validation_split=0.1,
    validation_batch_size=1000,  # examples are small, so we should be able to fit a lot in memory
)

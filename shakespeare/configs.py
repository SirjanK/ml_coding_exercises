from dataclasses import dataclass


@dataclass
class Config:
    """Configuration base class for model construction and training."""
    batch_size: int
    block_size: int
    embedding_size: int
    num_layers: int
    num_heads: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    validation_split: float
    validation_batch_size: int


# Lite config for testing out model training - meant for training on CPU (e.g. on laptop)
LITE_CONFIG = Config(
    batch_size=64,
    block_size=32,
    embedding_size=64,
    num_layers=2,
    num_heads=4,
    learning_rate=1e-3,
    max_iters=50000,
    eval_interval=500,
    validation_split=0.1,
    validation_batch_size=5000,
)

# Heavy config for model training on GPU
HEAVY_CONFIG = Config(
    batch_size=64,
    block_size=256,
    embedding_size=256,
    num_layers=4,
    num_heads=4,
    learning_rate=1e-4,
    max_iters=50000,
    eval_interval=500,
    validation_split=0.1,
    validation_batch_size=5000,
)

from shakespeare.train import train
from shakespeare.configs import Config


def test_train_loop():
    test_config = Config(
        batch_size=2,
        block_size=8,
        embedding_size=32,
        num_layers=2,
        num_heads=4,
        learning_rate=0.001,
        max_iters=10,
        eval_interval=2,
        validation_split=0.1,
        validation_batch_size=2,
    )

    # Run the training loop
    train(
        data_path="shakespeare/data/",
        config=test_config,
    )

class GeometricTrainConfig:
    random_seed: int = 42
    epochs: int = 50
    model_name: str
    hidden_dim: int = 128
    img_size: int = (512, 512)
    lr: float = 2e-4
    weight_decay: float = 1e-4
    k_values: list[int] = [60]
    batch_size: int = 128

    train_workers = 2
    val_workers = 2

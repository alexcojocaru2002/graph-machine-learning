from config.geometric_train_config import GeometricTrainConfig
from models.gat_geometric import MultiheadGAT
from training.train_geometric import geometric_training_entrypoint


if __name__ == "__main__":
    config = GeometricTrainConfig()
    config.model_name = "gat_k60"
    config.batch_size = 128
    config.epochs = 20

    gat_model = MultiheadGAT(in_dim=1024, hidden_dim=512, out_dim=6)

    geometric_training_entrypoint(gat_model, config)
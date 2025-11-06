from config.geometric_train_config import GeometricTrainConfig
from models.mlp import MLP
from training.train_geometric import geometric_training_entrypoint


if __name__ == "__main__":
    config = GeometricTrainConfig()
    config.model_name = "mlp_k60"

    mlp_model = MLP(in_dim=1024, hidden_dim=config.hidden_dim, out_dim=6)

    geometric_training_entrypoint(mlp_model, config)
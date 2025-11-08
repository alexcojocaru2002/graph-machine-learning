from config.geometric_train_config import GeometricTrainConfig
from models.transformer import Transformer
from training.train_geometric import geometric_training_entrypoint


if __name__ == "__main__":
    config = GeometricTrainConfig()
    config.model_name = "transformer_k60"
    config.epochs = 100

    mlp_model = Transformer(feature_dim=1024, hidden_dim=config.hidden_dim, out_dim=6)

    geometric_training_entrypoint(mlp_model, config)
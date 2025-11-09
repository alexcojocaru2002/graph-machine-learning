from config.geometric_train_config import GeometricTrainConfig
from models.gat_geometric import MultiheadGAT
from explanation.explain_geometric import geometric_explainer_entrypoint


if __name__ == "__main__":
    config = GeometricTrainConfig()
    config.model_name = "gat_k60"
    config.k_values = [60]
    config.batch_size = 32
    config.epochs = 100

    gat_model = MultiheadGAT(in_dim=1024, hidden_dim=512, out_dim=6)
    geometric_explainer_entrypoint(model = gat_model, config = config)

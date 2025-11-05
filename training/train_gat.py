from config.geometric_train_config import GeometricTrainConfig
from models.gat_geometric import MultiheadGAT
from training.train_geometric import geometric_training_entrypoint

config = GeometricTrainConfig()
config.model_name = "gat_k60"

gat_model = MultiheadGAT(in_dim=1024, hidden_dim=512, out_dim=6)

geometric_training_entrypoint(gat_model, config)
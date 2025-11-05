from config.geometric_train_config import GeometricTrainConfig
from models.gcn import GCN2
from training.train_geometric import geometric_training_entrypoint

config = GeometricTrainConfig()
config.model_name = "gcn2_k60"

gcn_model = GCN2(in_dim=1024, hidden_dim=config.hidden_dim, out_dim=6)

geometric_training_entrypoint(gcn_model, config)
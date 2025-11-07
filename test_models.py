from config.geometric_train_config import GeometricTrainConfig
from models.gcn import GCN2
from models.gat_geometric import MultiheadGAT
from training.train_geometric import test_model
import pandas as pd
import const

if __name__ == "__main__":
    configs = []

    # cfg_gcn = GeometricTrainConfig()
    # cfg_gcn.model_name = "gcn2_k60"
    # gcn_model = GCN2(in_dim=1024, hidden_dim=cfg_gcn.hidden_dim, out_dim=6)
    # configs.append((gcn_model, cfg_gcn))

    cfg_gat = GeometricTrainConfig()
    cfg_gat.model_name = "gat_k60"
    gat_model = MultiheadGAT(in_dim=1024, hidden_dim=512, out_dim=6)
    configs.append((gat_model, cfg_gat))

    results = [test_model(model, cfg) for model, cfg in configs]
    summary_path = const.ARTIFACTS_DIR / "test_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\nSaved all results to {summary_path}")
from config.geometric_train_config import GeometricTrainConfig
from models.gcn import GCN2
from models.gat_geometric import MultiheadGAT
from explanation.explain_geometric import geometric_explainer_entrypoint


def run_all_models():
    models = [
        (GCN2, "gcn2_k60", [60], None),
        (GCN2, "gcn2_k60_k120_k300_k500_multi", [60, 120, 300, 500], None),
        (MultiheadGAT, "gat_k60", [60], 512),
        (MultiheadGAT, "gat_k60_k120_k300_k500_multi", [60, 120, 300, 500], 512),
    ]

    for model_class, model_name, k_values, hidden_dim_override in models:
        print(f"\n[INFO] Running explanation for {model_name} ...")

        config = GeometricTrainConfig()
        config.model_name = model_name
        config.k_values = k_values

        if model_class == MultiheadGAT:
            config.batch_size = 32
            config.epochs = 100

        hidden_dim = hidden_dim_override or config.hidden_dim
        model = model_class(in_dim=1024, hidden_dim=hidden_dim, out_dim=6)

        geometric_explainer_entrypoint(model=model, config=config)


if __name__ == "__main__":
    run_all_models()

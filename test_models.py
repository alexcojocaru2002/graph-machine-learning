import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config.geometric_train_config import GeometricTrainConfig
import const
from models.gat_geometric import MultiheadGAT
from models.transformer import Transformer
from training.train_geometric import test_model  # assuming test_model is defined elsewhere

if __name__ == "__main__":
    configs = []

    # ---------------- Transformer ----------------
    cfg_trans = GeometricTrainConfig()
    cfg_trans.model_name = "transformer_k60_k120_k300_k500"
    transformer_model = Transformer(feature_dim=1024, hidden_dim=cfg_trans.hidden_dim, out_dim=6)
    configs.append((transformer_model, cfg_trans))

    # ---------------- GAT ----------------
    cfg_gat = GeometricTrainConfig()
    cfg_gat.model_name = "gat_k60_k120_k300_k500_multi"
    cfg_gat.k_values = [60, 120, 300, 500]  # 👈 correct place for k
    gat_model = MultiheadGAT(in_dim=1024, hidden_dim=512, out_dim=6)
    configs.append((gat_model, cfg_gat))

    # ----------------------------------------------------------------------
    # PART 1: Plot validation curves from saved training metrics
    # ----------------------------------------------------------------------
    for _, cfg in configs:
        model_name = cfg.model_name
        metrics_path = const.ARTIFACTS_DIR / f"{model_name}_metrics.csv"
        if not metrics_path.exists():
            print(f"⚠️ No metrics file found for {model_name}: {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        plt.figure(figsize=(8, 4))
        plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
        plt.plot(df["epoch"], df["val_mse"], marker="o", label="Validation MSE")
        plt.title(f"{model_name} — Validation Curves During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plot_path = const.ARTIFACTS_DIR / f"{model_name}_training_val_curves.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Saved training validation curves for {model_name} → {plot_path}")

    # ----------------------------------------------------------------------
    # PART 2: Evaluate models on the validation (test) split and compare
    # ----------------------------------------------------------------------
    results = []
    for model, cfg in configs:
        model_name = cfg.model_name
        ckpt_path = const.ARTIFACTS_DIR / f"{model_name}_best.ckpt"
        if not ckpt_path.exists():
            print(f"⚠️ Checkpoint not found for {model_name}: {ckpt_path}")
            continue

        # load model weights properly
        if isinstance(model, MultiheadGAT):
            model = MultiheadGAT.load_model(cfg, num_classes_eff=6)
        elif isinstance(model, Transformer):
            model = Transformer.load_model(cfg, num_classes_eff=6)
        else:
            print(f"⚠️ Unknown model type for {model_name}, skipping.")
            continue

        # run evaluation using your existing test_model()
        results.append(test_model(model, cfg))

    summary_path = const.ARTIFACTS_DIR / "test_summary.csv"
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✅ Saved validation/test summary to {summary_path}")

    # ----------------------------------------------------------------------
    # PART 3: Plot histogram comparing models’ validation performance
    # ----------------------------------------------------------------------
    if not df_summary.empty:
        plt.figure(figsize=(7, 5))
        plt.bar(df_summary["model"], df_summary["test_mse"],
                color="skyblue", edgecolor="black")
        plt.title("Model Performance on Validation (Test) Set")
        plt.ylabel("Validation MSE")
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Annotate each bar
        for i, val in enumerate(df_summary["test_mse"]):
            plt.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        mse_plot_path = const.ARTIFACTS_DIR / "val_mse_comparison.png"
        plt.savefig(mse_plot_path)
        plt.show()
        print(f"📊 Saved validation MSE comparison histogram → {mse_plot_path}")
    else:
        print("⚠️ No results available to plot.")

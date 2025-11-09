import torch
from config.geometric_train_config import GeometricTrainConfig
from models.gat_geometric import MultiheadGAT
from training.train_geometric import geometric_training_entrypoint


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Number of devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available.")

    config = GeometricTrainConfig()
    config.model_name = "gat_k60_k120_k300_k500_multi"
    config.batch_size = 32
    config.epochs = 100
    config.k_values = [60, 120, 300, 500]

    gat_model = MultiheadGAT(in_dim=1024, hidden_dim=512, out_dim=6)

    geometric_training_entrypoint(gat_model, config)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Project input features to hidden_dim if needed
        self.input_proj = nn.Linear(feature_dim, hidden_dim) if feature_dim != hidden_dim else nn.Identity()
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer model.
        
        Args:
            x: Node features of shape (num_nodes, feature_dim)
        
        Returns:
            Node predictions of shape (num_nodes, out_dim)
        """
        # Add batch dimension if needed (num_nodes, feature_dim) -> (1, num_nodes, feature_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Project to hidden_dim
        x = self.input_proj(x)  # (batch, num_nodes, hidden_dim)
        
        # Apply transformer
        x = self.transformer(x)  # (batch, num_nodes, hidden_dim)
        
        # Project to output dimension
        x = self.output_proj(x)  # (batch, num_nodes, out_dim)
        
        # Remove batch dimension if we added it
        if x.size(0) == 1:
            x = x.squeeze(0)  # (num_nodes, out_dim)
        
        return x

    def get_architecture(self) -> dict:
        """Return a JSON-serializable dict describing the model architecture."""
        return {
            "model_class": self.__class__.__name__,
            "feature_dim": int(self.feature_dim),
            "hidden_dim": int(self.hidden_dim),
            "out_dim": int(self.out_dim),
            "dropout": float(self.dropout),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "final_activation": "softmax",
        }

    @staticmethod
    def load_model(cfg, num_classes_eff: int) -> "MLP":
        """
        Build transformer model based on checkpoint architecture metadata if available,
        otherwise fall back to values in cfg. Also loads checkpoint weights if provided.
        """
        from pathlib import Path
        import json

        device = torch.device(getattr(cfg, "device", "cpu"))
        arch_path = Path(cfg.ckpt_path).with_suffix(".json") if getattr(cfg, "ckpt_path", None) else None
        arch = None
        if arch_path is not None and arch_path.exists():
            with open(arch_path, "r") as f:
                meta = json.load(f)
            arch = meta.get("architecture", None)

        if arch is not None:
            out_dim = arch.get("out_dim", num_classes_eff)
            if out_dim != num_classes_eff:
                out_dim = num_classes_eff
            model = Transformer(
                feature_dim=arch.get("feature_dim", arch.get("in_dim", 1024)),  # backward compat
                hidden_dim=arch.get("hidden_dim", getattr(cfg, "hidden_dim", 512)),
                out_dim=out_dim,
                dropout=arch.get("dropout", getattr(cfg, "dropout", 0.2)),
                num_layers=arch.get("num_layers", getattr(cfg, "num_layers", 2)),
                num_heads=arch.get("num_heads", getattr(cfg, "num_heads", 4)),
            ).to(device)
        else:
            model = Transformer(
                feature_dim=getattr(cfg, "feature_dim", getattr(cfg, "in_dim", 1024)),  # backward compat
                hidden_dim=getattr(cfg, "hidden_dim", 512),
                out_dim=num_classes_eff,
                dropout=getattr(cfg, "dropout", 0.2),
                num_layers=getattr(cfg, "num_layers", 2),
                num_heads=getattr(cfg, "num_heads", 4),
            ).to(device)

        ckpt_path = getattr(cfg, "ckpt_path", None)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt  
            model.load_state_dict(state_dict, strict=True)

        model.eval()
        return model

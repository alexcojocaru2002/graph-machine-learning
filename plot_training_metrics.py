import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

import const


def plot_metrics(csv_path):
    """Plot training metrics from a CSV file."""
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded metrics from {csv_path}")
    print(f"Epochs: {df['epoch'].min()} to {df['epoch'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (KL Divergence)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation MSE
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['val_mse'], marker='o', color='red', linewidth=2, label='Val MSE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Validation MSE', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(csv_path).with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Best Val Loss: {df['val_loss'].min():.6f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']:.0f})")
    print(f"  Best Val MSE:  {df['val_mse'].min():.6f} (Epoch {df.loc[df['val_mse'].idxmin(), 'epoch']:.0f})")
    print(f"  Final Train Loss: {df['train_loss'].iloc[-1]:.6f}")
    print(f"  Final Val Loss:   {df['val_loss'].iloc[-1]:.6f}")
    print(f"  Final Val MSE:    {df['val_mse'].iloc[-1]:.6f}")
    
    plt.show()


def main():
    # Default path
    default_csv = const.ARTIFACTS_DIR / "gat_k60_metrics.csv"
    
    # Allow command line argument to override
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = default_csv
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        print(f"\nUsage: python plot_training_metrics.py [path/to/metrics.csv]")
        print(f"Default: {default_csv}")
        sys.exit(1)
    
    plot_metrics(csv_path)


if __name__ == "__main__":
    main()

"""
Analyze training loss curves from saved models.

Usage:
    python analyze_loss.py --run-id gtcz
    python analyze_loss.py --run-id gtcz --save-plot
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_history(run_id):
    """Load training history from JSON file."""
    history_path = f'saved/{run_id}/training_history.json'
    if not Path(history_path).exists():
        print(f"Error: Training history not found at {history_path}")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    return history


def analyze_loss_curve(history):
    """Analyze and print statistics about the loss curve."""
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss']) if history['val_loss'] else None
    epochs = history['epochs']

    print("="*80)
    print("LOSS CURVE ANALYSIS")
    print("="*80)

    print(f"\nTotal epochs: {len(epochs)}")

    # Training loss statistics
    print(f"\nTraining Loss:")
    print(f"  Initial: {train_loss[0]:.6f}")
    print(f"  Final: {train_loss[-1]:.6f}")
    print(f"  Reduction: {(train_loss[0] - train_loss[-1]):.6f} ({(1 - train_loss[-1]/train_loss[0])*100:.2f}%)")
    print(f"  Min: {np.min(train_loss):.6f} (epoch {np.argmin(train_loss)+1})")
    print(f"  Mean: {np.mean(train_loss):.6f}")
    print(f"  Std: {np.std(train_loss):.6f}")

    if val_loss is not None:
        print(f"\nValidation Loss:")
        print(f"  Initial: {val_loss[0]:.6f}")
        print(f"  Final: {val_loss[-1]:.6f}")
        print(f"  Reduction: {(val_loss[0] - val_loss[-1]):.6f} ({(1 - val_loss[-1]/val_loss[0])*100:.2f}%)")
        print(f"  Min: {np.min(val_loss):.6f} (epoch {np.argmin(val_loss)+1})")
        print(f"  Mean: {np.mean(val_loss):.6f}")
        print(f"  Std: {np.std(val_loss):.6f}")

        # Check for overfitting
        gap = val_loss[-1] - train_loss[-1]
        print(f"\nOverfitting Analysis:")
        print(f"  Train-Val Gap (final): {gap:.6f}")
        if gap < 0.001:
            print(f"  Status: No significant overfitting")
        elif gap < 0.003:
            print(f"  Status: Slight overfitting")
        else:
            print(f"  Status: Moderate overfitting detected")

    # Convergence analysis
    print(f"\nConvergence Analysis:")
    last_5_train = train_loss[-5:]
    train_variance = np.var(last_5_train)
    print(f"  Last 5 epochs train loss variance: {train_variance:.8f}")
    if train_variance < 0.00001:
        print(f"  Status: Converged (low variance)")
    elif train_variance < 0.0001:
        print(f"  Status: Near convergence")
    else:
        print(f"  Status: Still decreasing")

    # Loss reduction per epoch (average)
    avg_reduction = (train_loss[0] - train_loss[-1]) / len(epochs)
    print(f"  Average loss reduction per epoch: {avg_reduction:.8f}")

    return train_loss, val_loss, epochs


def plot_loss_curve(train_loss, val_loss, epochs, run_id, save_plot=False):
    """Plot the loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Full loss curve
    axes[0].plot(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
    if val_loss is not None:
        axes[0].plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Reconstruction Error', fontsize=12)
    axes[0].set_title(f'Training Progress - Run ID: {run_id}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Log scale
    axes[1].semilogy(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
    if val_loss is not None:
        axes[1].semilogy(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Reconstruction Error (log scale)', fontsize=12)
    axes[1].set_title('Training Progress (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_plot:
        output_path = f'saved/{run_id}/loss_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main(args):
    # Load history
    print(f"Loading training history for run ID: {args.run_id}")
    history = load_training_history(args.run_id)

    if history is None:
        return

    # Analyze
    train_loss, val_loss, epochs = analyze_loss_curve(history)

    # Plot
    print("\n" + "="*80)
    print("PLOTTING")
    print("="*80)
    plot_loss_curve(train_loss, val_loss, epochs, args.run_id, save_plot=args.save_plot)

    # Load config for context
    config_path = f'saved/{args.run_id}/config.json'
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"Hidden units: {config['n_hidden']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"CD steps: {config['cd_steps']}")
        print(f"Ingredients: {config['n_ingredients']}")
        print(f"Train samples: {config['n_train_samples']:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze RBM training loss curves')

    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID to analyze (e.g., gtcz)')
    parser.add_argument('--save-plot', action='store_true',
                       help='Save plot instead of displaying')

    args = parser.parse_args()
    main(args)

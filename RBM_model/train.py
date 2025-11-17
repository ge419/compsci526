import numpy as np
import json
import os
import argparse
import random
import string
import matplotlib.pyplot as plt
from dataset import download_mm_food_100k
from model.rbm import BernoulliRBM
from model.ingredient_graph import parse_ingredients


def generate_run_id(length=4):
    """
    Generate a random alphanumeric run ID.

    Args:
        length: Length of the run ID (default: 4)

    Returns:
        Random string of lowercase letters
    """
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def prepare_ingredient_data(df, min_ingredient_count=10):
    """
    Prepare binary ingredient matrix from dataset.

    Args:
        df: DataFrame containing the dataset
        min_ingredient_count: Minimum number of occurrences for an ingredient to be included

    Returns:
        X: Binary matrix (n_samples, n_ingredients)
        ingredients: List of ingredient names
        ingredient_to_idx: Mapping from ingredient name to index
    """
    print("Preparing ingredient data...")

    # Extract all ingredients
    all_ingredients = []
    ingredient_lists = []

    for ingredients in df['ingredients']:
        ingredient_list = parse_ingredients(ingredients)
        ingredient_lists.append(ingredient_list)
        all_ingredients.extend(ingredient_list)

    # Count ingredient frequencies
    from collections import Counter
    ingredient_counts = Counter(all_ingredients)

    # Filter ingredients by minimum count
    filtered_ingredients = sorted([ing for ing, count in ingredient_counts.items()
                                  if count >= min_ingredient_count])

    print(f"Total unique ingredients: {len(ingredient_counts)}")
    print(f"Ingredients with >= {min_ingredient_count} occurrences: {len(filtered_ingredients)}")

    # Create ingredient to index mapping
    ingredient_to_idx = {ing: idx for idx, ing in enumerate(filtered_ingredients)}

    # Create binary matrix
    n_samples = len(ingredient_lists)
    n_ingredients = len(filtered_ingredients)
    X = np.zeros((n_samples, n_ingredients), dtype=np.float32)

    for i, ingredient_list in enumerate(ingredient_lists):
        for ing in ingredient_list:
            if ing in ingredient_to_idx:
                idx = ingredient_to_idx[ing]
                X[i, idx] = 1.0

    print(f"Binary matrix shape: {X.shape}")
    print(f"Sparsity: {1 - np.count_nonzero(X) / X.size:.4f}")
    print(f"Average ingredients per dish: {np.mean(np.sum(X, axis=1)):.2f}")

    return X, filtered_ingredients, ingredient_to_idx


def split_data(X, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets.

    Args:
        X: Full dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation

    Returns:
        X_train, X_val, X_test
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    return X_train, X_val, X_test


def plot_training_history(history, save_path='fig/training_history.png'):
    """
    Plot training and validation loss.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)

    if history['val_loss'] is not None:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('RBM Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}")


def test_generation(rbm, ingredients, n_samples=5):
    """
    Test the RBM by generating sample ingredient combinations.

    Args:
        rbm: Trained RBM model
        ingredients: List of ingredient names
        n_samples: Number of samples to generate
    """
    print("\n" + "="*60)
    print("GENERATING SAMPLE INGREDIENT COMBINATIONS")
    print("="*60)

    # Generate samples
    generated = rbm.generate(n_samples=n_samples, n_gibbs=1000)

    for i in range(n_samples):
        sample = generated[i]
        top_ingredients = rbm.get_top_ingredients(sample, top_k=10)

        print(f"\nSample {i+1}:")
        if top_ingredients:
            for ing, prob in top_ingredients:
                print(f"  - {ing} ({prob:.2f})")
        else:
            print("  No ingredients generated (threshold too high)")


def test_conditional_generation(rbm, ingredients, seed_ingredients_list):
    """
    Test conditional generation given seed ingredients.

    Args:
        rbm: Trained RBM model
        ingredients: List of ingredient names
        seed_ingredients_list: List of seed ingredient combinations
    """
    print("\n" + "="*60)
    print("CONDITIONAL INGREDIENT GENERATION")
    print("="*60)

    for seed_ingredients in seed_ingredients_list:
        print(f"\nSeed ingredients: {', '.join(seed_ingredients)}")

        # Generate
        generated = rbm.generate(n_samples=1, n_gibbs=1000, seed_ingredients=seed_ingredients)
        sample = generated[0]

        # Get top ingredients
        top_ingredients = rbm.get_top_ingredients(sample, top_k=15)

        print("Generated combination:")
        if top_ingredients:
            for ing, prob in top_ingredients:
                # Mark seed ingredients
                marker = "[SEED]" if ing in seed_ingredients else ""
                print(f"  - {ing} ({prob:.2f}) {marker}")
        else:
            print("  No ingredients generated")


def main(args):
    """Main training function."""
    # Generate run ID
    run_id = generate_run_id()

    print("="*60)
    print("RBM TRAINING FOR INGREDIENT GENERATION")
    print("="*60)
    print(f"\nRun ID: {run_id}")

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Load dataset
    print("\nLoading dataset...")
    dataset = download_mm_food_100k()
    df = dataset['train'].to_pandas()
    print(f"Loaded {len(df)} dishes")

    # Prepare data
    X, ingredients, ingredient_to_idx = prepare_ingredient_data(
        df, min_ingredient_count=args.min_count
    )

    # Split data
    X_train, X_val, X_test = split_data(X, train_ratio=0.8, val_ratio=0.1)

    # Initialize RBM
    print("\n" + "="*60)
    print("INITIALIZING RBM")
    print("="*60)
    rbm = BernoulliRBM(
        n_visible=X.shape[1],
        n_hidden=args.n_hidden,
        learning_rate=args.learning_rate,
        ingredients=ingredients
    )
    print(rbm)

    # Train RBM
    print("\n" + "="*60)
    print("TRAINING RBM")
    print("="*60)
    history = rbm.fit(
        X_train,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        k=args.cd_steps,
        verbose=True,
        validation_data=X_val
    )

    # Evaluate on test set
    test_loss = rbm.reconstruction_error(X_test)
    print(f"\nTest reconstruction error: {test_loss:.4f}")

    # Create run directory
    run_dir = f'saved/{run_id}'
    os.makedirs(run_dir, exist_ok=True)

    # Plot training history
    plot_path = f'{run_dir}/training_history.png'
    plot_training_history(history, save_path=plot_path)

    # Save model
    model_path = f'{run_dir}/model.npz'
    rbm.save(model_path)

    # Save training configuration
    config = {
        'run_id': run_id,
        'n_visible': X.shape[1],
        'n_hidden': args.n_hidden,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'cd_steps': args.cd_steps,
        'min_ingredient_count': args.min_count,
        'test_loss': float(test_loss),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]) if history['val_loss'] else None,
        'n_ingredients': len(ingredients),
        'n_train_samples': X_train.shape[0],
        'n_val_samples': X_val.shape[0],
        'n_test_samples': X_test.shape[0],
        'random_seed': args.seed
    }
    config_path = f'{run_dir}/config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")

    # Save training history as JSON
    history_dict = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']] if history['val_loss'] else None,
        'epochs': list(range(1, len(history['train_loss']) + 1))
    }
    history_path = f'{run_dir}/training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Test generation
    if args.test_generation:
        test_generation(rbm, ingredients, n_samples=5)

        # Test conditional generation
        seed_examples = [
            ['chicken', 'rice'],
            ['beef', 'vegetables'],
            ['noodles', 'broth'],
            ['flour', 'sugar', 'egg'],
        ]
        test_conditional_generation(rbm, ingredients, seed_examples)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nRun ID: {run_id}")
    print(f"Run directory: {run_dir}/")
    print(f"\nSaved files:")
    print(f"  - Model: {model_path}")
    print(f"  - Training plot: {plot_path}")
    print(f"  - Training history (JSON): {history_path}")
    print(f"  - Configuration: {config_path}")
    print(f"\nTo load this model:")
    print(f"  from model.rbm import BernoulliRBM")
    print(f"  rbm = BernoulliRBM.load_model('{model_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RBM for ingredient generation')

    # Data parameters
    parser.add_argument('--min-count', type=int, default=50,
                       help='Minimum ingredient occurrence count (default: 50)')

    # Model parameters
    parser.add_argument('--n-hidden', type=int, default=500,
                       help='Number of hidden units (default: 500)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')

    # Training parameters
    parser.add_argument('--n-epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--cd-steps', type=int, default=1,
                       help='Number of Contrastive Divergence steps (default: 1)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--test-generation', action='store_true',
                       help='Test generation after training')

    args = parser.parse_args()
    main(args)

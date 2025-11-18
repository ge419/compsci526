"""
Test script to verify minimum ingredient filtering works.
Generates multiple recipes and checks that all have at least min_ingredients.
"""

import argparse
import numpy as np
from model.rbm import BernoulliRBM

def test_min_ingredients(run_id: str, min_ingredients: int = 3, n_tests: int = 10):
    """Test that generated recipes have at least min_ingredients."""
    print(f"\n{'='*80}")
    print(f"Testing minimum ingredient requirement: {min_ingredients}")
    print(f"{'='*80}\n")

    # Load model
    model_path = f'saved/{run_id}/model.npz'
    rbm = BernoulliRBM.load_model(model_path)

    # Initialize user preference
    user_pref = rbm.init_user_preference(latent_dim=50)

    # Stats
    total_generations = 0
    accepted = 0
    rejected = 0
    ingredient_counts = []

    print(f"Generating {n_tests} valid recipes...\n")

    while accepted < n_tests and total_generations < n_tests * 20:  # Max 20x attempts
        total_generations += 1

        # Generate
        generated = rbm.generate_with_preference(
            user_pref,
            n_samples=1,
            n_gibbs=1000,
            alpha=0.5
        )

        # Get ingredients
        sample = generated[0]
        top_ingredients = rbm.get_top_ingredients(sample, top_k=10)
        ingredient_names = [ing for ing, _ in top_ingredients]
        count = len(ingredient_names)

        if count < min_ingredients:
            rejected += 1
            print(f"  Generation {total_generations}: {count} ingredient(s) - REJECTED")
        else:
            accepted += 1
            ingredient_counts.append(count)
            print(f"  Generation {total_generations}: {count} ingredients - ACCEPTED ({', '.join(ingredient_names[:5])}...)")

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Minimum required: {min_ingredients} ingredients")
    print(f"Total generations: {total_generations}")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected} (too few ingredients)")
    print(f"Rejection rate: {rejected/total_generations*100:.1f}%")

    if ingredient_counts:
        print(f"\nAccepted recipe stats:")
        print(f"  Average ingredients: {np.mean(ingredient_counts):.1f}")
        print(f"  Min ingredients: {min(ingredient_counts)}")
        print(f"  Max ingredients: {max(ingredient_counts)}")

    # Verify all accepted recipes meet requirement
    violations = [c for c in ingredient_counts if c < min_ingredients]
    if violations:
        print(f"\n❌ FAILED: {len(violations)} accepted recipes had fewer than {min_ingredients} ingredients")
        return False
    else:
        print(f"\n✅ SUCCESS: All accepted recipes have at least {min_ingredients} ingredients")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test minimum ingredient filtering')
    parser.add_argument('--run-id', type=str, default='gtcz', help='Run ID to test')
    parser.add_argument('--min-ingredients', type=int, default=3,
                       help='Minimum ingredients required')
    parser.add_argument('--n-tests', type=int, default=10,
                       help='Number of valid recipes to generate')

    args = parser.parse_args()

    success = test_min_ingredients(args.run_id, args.min_ingredients, args.n_tests)
    exit(0 if success else 1)

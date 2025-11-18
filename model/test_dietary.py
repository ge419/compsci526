"""
Test script to verify dietary restrictions work correctly.
Generate a few recipes and check that ingredients are appropriate.
"""

import argparse
import numpy as np
import json
from pathlib import Path
from model.rbm import BernoulliRBM

def load_dietary_restrictions():
    """Load the dietary restrictions JSON."""
    with open('dietary_restrictions.json') as f:
        return json.load(f)

def test_dietary_restriction(run_id: str, restriction: str, n_tests: int = 5):
    """Test that generated ingredients respect dietary restrictions."""
    print(f"\n{'='*80}")
    print(f"Testing dietary restriction: {restriction}")
    print(f"{'='*80}\n")

    # Load model
    model_path = f'saved/{run_id}/model.npz'
    rbm = BernoulliRBM.load_model(model_path)

    # Load dietary restrictions
    dietary_data = load_dietary_restrictions()

    # Create ingredient mask if needed
    ingredient_mask = None
    if restriction in ['vegan', 'vegetarian']:
        ingredient_mask = np.ones(rbm.n_visible, dtype=np.float32)
        for idx, ing_name in enumerate(rbm.ingredients):
            ing_lower = ing_name.lower()
            if ing_lower in dietary_data:
                restrictions = dietary_data[ing_lower]['restricted_for']
                if restriction in restrictions:
                    ingredient_mask[idx] = 0.0

        forbidden_count = int(np.sum(ingredient_mask == 0))
        print(f"Ingredient mask created: {forbidden_count} forbidden ingredients")

    # Initialize user preference
    user_pref = rbm.init_user_preference(latent_dim=50)

    # Generate and check multiple samples
    violations = []
    for i in range(n_tests):
        print(f"\n--- Test {i+1}/{n_tests} ---")

        # Generate
        generated = rbm.generate_with_preference(
            user_pref,
            n_samples=1,
            n_gibbs=1000,
            alpha=0.5,
            ingredient_mask=ingredient_mask
        )

        # Get ingredients
        sample = generated[0]
        top_ingredients = rbm.get_top_ingredients(sample, top_k=10)
        ingredient_names = [ing for ing, _ in top_ingredients]

        print(f"Generated ingredients: {', '.join(ingredient_names)}")

        # Check for violations
        if restriction in ['vegan', 'vegetarian']:
            for ing in ingredient_names:
                ing_lower = ing.lower()
                if ing_lower in dietary_data:
                    restricted_for = dietary_data[ing_lower]['restricted_for']
                    if restriction in restricted_for:
                        violations.append({
                            'test': i+1,
                            'ingredient': ing,
                            'type': dietary_data[ing_lower]
                        })
                        print(f"  ⚠️  VIOLATION: {ing} is forbidden for {restriction}")

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Restriction: {restriction}")
    print(f"Tests run: {n_tests}")
    print(f"Violations: {len(violations)}")

    if violations:
        print("\nViolation details:")
        for v in violations:
            print(f"  Test {v['test']}: {v['ingredient']} - {v['type']}")
        print("\n❌ FAILED: Dietary restrictions not properly enforced")
        return False
    else:
        print("\n✅ SUCCESS: All generated recipes respect dietary restrictions")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test dietary restriction filtering')
    parser.add_argument('--run-id', type=str, default='gtcz', help='Run ID to test')
    parser.add_argument('--restriction', type=str, default='vegan',
                       choices=['none', 'vegetarian', 'vegan'],
                       help='Dietary restriction to test')
    parser.add_argument('--n-tests', type=int, default=5,
                       help='Number of test generations')

    args = parser.parse_args()

    success = test_dietary_restriction(args.run_id, args.restriction, args.n_tests)
    exit(0 if success else 1)

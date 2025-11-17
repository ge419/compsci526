"""
Script to generate ingredient combinations using a trained RBM model and find matching dishes.

Usage:
    python generate.py --run-id dwma --n-samples 5
    python generate.py --run-id dwma --n-samples 3 --show-dishes --top-k 3
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from model.rbm import BernoulliRBM
from dataset import download_mm_food_100k
from model.ingredient_graph import parse_ingredients


class DishMatcher:
    """Match generated ingredient lists to actual dishes in the dataset."""

    def __init__(self, df):
        """Initialize the dish matcher."""
        self.df = df
        self._preprocess_ingredients()

    def _preprocess_ingredients(self):
        """Parse all ingredients in the dataset."""
        self.df['ingredient_list'] = self.df['ingredients'].apply(parse_ingredients)
        self.df['ingredient_set'] = self.df['ingredient_list'].apply(set)

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def search_dishes(self, generated_ingredients: List[str], top_k: int = 5) -> List[Dict]:
        """Search for dishes that best match the generated ingredients."""
        generated_set = set(generated_ingredients)

        # Compute similarity for all dishes
        similarities = []
        for idx, row in self.df.iterrows():
            dish_ingredients = row['ingredient_set']
            score = self.jaccard_similarity(generated_set, dish_ingredients)

            similarities.append({
                'index': idx,
                'score': score,
                'dish_name': row['dish_name'],
                'ingredients': row['ingredient_list'],
                'food_type': row['food_type'],
                'cooking_method': row['cooking_method'],
                'image_url': row['image_url'],
                'nutritional_profile': row['nutritional_profile'],
            })

        # Sort by score (descending)
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]


def display_dish_matches(matches: List[Dict], generated_ingredients: List[str]):
    """Display matching dishes in a readable format."""
    print("\n  " + "-"*76)
    print("  MATCHING DISHES")
    print("  " + "-"*76)

    for i, match in enumerate(matches, 1):
        print(f"\n  {i}. {match['dish_name']} (Score: {match['score']:.3f})")
        print(f"     Food Type: {match['food_type']}")
        if match['cooking_method']:
            print(f"     Cooking: {match['cooking_method']}")

        # Show ingredient comparison
        generated_set = set(generated_ingredients)
        dish_set = set(match['ingredients'])
        matching = generated_set & dish_set

        if matching:
            print(f"     ✓ Match: {', '.join(sorted(matching))}")

        missing = generated_set - dish_set
        if missing and len(missing) <= 5:  # Only show if not too many
            print(f"     ✗ Missing: {', '.join(sorted(missing))}")

        # Parse nutrition
        try:
            nutrition = json.loads(match['nutritional_profile']) if isinstance(match['nutritional_profile'], str) else match['nutritional_profile']
            if nutrition:
                cal = nutrition.get('calories_kcal', nutrition.get('calories', 'N/A'))
                protein = nutrition.get('protein_g', nutrition.get('protein', 'N/A'))
                carbs = nutrition.get('carbohydrate_g', nutrition.get('carbohydrates', 'N/A'))
                fat = nutrition.get('fat_g', nutrition.get('fat', 'N/A'))
                print(f"     Nutrition: {cal} kcal | P: {protein}g | C: {carbs}g | F: {fat}g")
        except:
            pass

        print(f"     Image: {match['image_url']}")


def generate_random(rbm, n_samples=5, n_gibbs=1000, matcher=None, top_k_dishes=3):
    """Generate random ingredient combinations."""
    print("="*80)
    print(f"GENERATING {n_samples} RANDOM INGREDIENT COMBINATIONS")
    print("="*80)

    generated = rbm.generate(n_samples=n_samples, n_gibbs=n_gibbs)

    for i in range(n_samples):
        sample = generated[i]
        top_ingredients = rbm.get_top_ingredients(sample, top_k=10)

        print(f"\n{'='*80}")
        print(f"Sample {i+1}")
        print(f"{'='*80}")

        if top_ingredients:
            ingredient_names = [ing for ing, prob in top_ingredients]
            print(f"\nGenerated Ingredients ({len(ingredient_names)}):")
            for ing, prob in top_ingredients:
                print(f"  - {ing}")

            # Search for matching dishes
            if matcher:
                matches = matcher.search_dishes(ingredient_names, top_k=top_k_dishes)
                display_dish_matches(matches, ingredient_names)
        else:
            print("  No ingredients generated")


def main(args):
    # Set random seed only if explicitly provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}\n")

    # Determine model path from run ID
    model_path = f'saved/{args.run_id}/model.npz'
    if not Path(model_path).exists():
        print(f"Error: Run ID '{args.run_id}' not found")
        print(f"Expected model at: {model_path}")
        return

    # Load model
    print(f"Loading model from {model_path}...\n")
    rbm = BernoulliRBM.load_model(model_path)
    print()

    # Load dataset and create matcher if needed
    matcher = None
    if args.show_dishes:
        print("Loading dataset for dish matching...")
        dataset = download_mm_food_100k()
        df = dataset['train'].to_pandas()
        print("Preprocessing ingredients...")
        matcher = DishMatcher(df)
        print(f"Ready to match against {len(df):,} dishes\n")

    # Generate
    generate_random(rbm, n_samples=args.n_samples, n_gibbs=args.n_gibbs,
                   matcher=matcher, top_k_dishes=args.top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate ingredient combinations using trained RBM'
    )

    # Model specification
    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID to load (e.g., dwma)')

    # Generation parameters
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of samples to generate (default: 5)')
    parser.add_argument('--n-gibbs', type=int, default=1000,
                       help='Number of Gibbs sampling steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None, random each time)')

    # Dish matching parameters
    parser.add_argument('--show-dishes', action='store_true',
                       help='Show matching dishes from dataset')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top matching dishes to show (default: 3)')

    args = parser.parse_args()
    main(args)

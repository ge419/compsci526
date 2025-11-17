"""
Interactive dish recommendation with user preference learning.

The system:
1. Generates ingredient combinations based on your preferences
2. Shows you matching dishes from the dataset
3. Learns from your accept/reject feedback
4. Adapts recommendations over time

Usage:
    python interactive.py --run-id dwma
    python interactive.py --run-id dwma --latent-dim 100 --alpha 0.7
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


def display_dish(dish: Dict, rank: int, generated_ingredients: List[str]):
    """Display a single dish with details."""
    print(f"\n{'='*80}")
    print(f"DISH #{rank}")
    print(f"{'='*80}")
    print(f"Name: {dish['dish_name']}")
    print(f"Food Type: {dish['food_type']}")
    if dish['cooking_method']:
        print(f"Cooking Method: {dish['cooking_method']}")
    print(f"Match Score: {dish['score']:.3f}")

    # Show ingredient comparison
    generated_set = set(generated_ingredients)
    dish_set = set(dish['ingredients'])
    matching = generated_set & dish_set

    print(f"\nIngredients:")
    print(f"  Full list: {', '.join(dish['ingredients'])}")
    if matching:
        print(f"  ✓ Matching generated: {', '.join(sorted(matching))}")

    # Parse and display nutrition
    try:
        nutrition = json.loads(dish['nutritional_profile']) if isinstance(dish['nutritional_profile'], str) else dish['nutritional_profile']
        if nutrition:
            cal = nutrition.get('calories_kcal', nutrition.get('calories', 'N/A'))
            protein = nutrition.get('protein_g', nutrition.get('protein', 'N/A'))
            carbs = nutrition.get('carbohydrate_g', nutrition.get('carbohydrates', 'N/A'))
            fat = nutrition.get('fat_g', nutrition.get('fat', 'N/A'))
            print(f"\nNutrition: {cal} kcal | Protein: {protein}g | Carbs: {carbs}g | Fat: {fat}g")
    except:
        pass

    print(f"\nImage URL: {dish['image_url']}")


def interactive_session(rbm, matcher, args):
    """Run an interactive recommendation session."""
    print("="*80)
    print("INTERACTIVE DISH RECOMMENDATION SYSTEM")
    print("="*80)
    print("\nThis system learns your preferences as you accept or reject suggestions.")
    print("Type 'accept' or 'a' to accept a dish")
    print("Type 'reject' or 'r' to reject a dish")
    print("Type 'quit' or 'q' to exit")
    print("="*80)

    # Initialize user preference vector
    user_pref = rbm.init_user_preference(latent_dim=args.latent_dim)
    print(f"\nInitialized user preference vector (dimension: {args.latent_dim})")
    print(f"Preference strength (alpha): {args.alpha}")
    print(f"Learning rate: {args.learning_rate}\n")

    iteration = 0
    accepted_count = 0
    rejected_count = 0

    while True:
        iteration += 1
        print(f"\n{'#'*80}")
        print(f"ITERATION {iteration}")
        print(f"#'*80}")
        print(f"Stats: {accepted_count} accepted, {rejected_count} rejected")

        # Generate ingredient combination based on current preferences
        print("\nGenerating personalized recommendation...")
        generated = rbm.generate_with_preference(
            user_pref,
            n_samples=1,
            n_gibbs=args.n_gibbs,
            alpha=args.alpha
        )

        # Extract ingredients
        sample = generated[0]
        top_ingredients = rbm.get_top_ingredients(sample, top_k=10)
        ingredient_names = [ing for ing, _ in top_ingredients]

        print(f"\nGenerated Ingredients ({len(ingredient_names)}):")
        for ing, prob in top_ingredients:
            print(f"  - {ing}")

        # Find matching dishes
        print(f"\nSearching for top {args.top_k} matching dishes...")
        matches = matcher.search_dishes(ingredient_names, top_k=args.top_k)

        if not matches or matches[0]['score'] < 0.01:
            print("\nNo good matches found. Trying again...")
            continue

        # Show top dish
        top_dish = matches[0]
        display_dish(top_dish, 1, ingredient_names)

        # Show other options if requested
        if len(matches) > 1:
            print(f"\n(Type 'more' to see {len(matches)-1} other options)")

        # Get user feedback
        while True:
            try:
                feedback = input("\nYour feedback (accept/reject/more/quit): ").strip().lower()

                if feedback in ['quit', 'q']:
                    print(f"\n{'='*80}")
                    print("SESSION SUMMARY")
                    print(f"{'='*80}")
                    print(f"Total iterations: {iteration}")
                    print(f"Accepted: {accepted_count}")
                    print(f"Rejected: {rejected_count}")
                    print(f"\nFinal preference vector norm: {np.linalg.norm(user_pref):.3f}")
                    print("\nThanks for using the recommendation system!")
                    return

                elif feedback == 'more':
                    # Show remaining dishes
                    for i, dish in enumerate(matches[1:], start=2):
                        display_dish(dish, i, ingredient_names)
                    continue

                elif feedback in ['accept', 'a']:
                    print("\n✓ Accepted! Updating your preferences...")
                    user_pref = rbm.update_user_preference(
                        user_pref,
                        ingredient_names,
                        'accept',
                        learning_rate=args.learning_rate
                    )
                    accepted_count += 1
                    print(f"Preference vector norm: {np.linalg.norm(user_pref):.3f}")
                    break

                elif feedback in ['reject', 'r']:
                    print("\n✗ Rejected! Updating your preferences...")
                    user_pref = rbm.update_user_preference(
                        user_pref,
                        ingredient_names,
                        'reject',
                        learning_rate=args.learning_rate
                    )
                    rejected_count += 1
                    print(f"Preference vector norm: {np.linalg.norm(user_pref):.3f}")
                    break

                else:
                    print("Invalid input. Please type 'accept', 'reject', 'more', or 'quit'.")

            except (EOFError, KeyboardInterrupt):
                print("\n\nSession interrupted.")
                return


def main(args):
    # Load model
    model_path = f'saved/{args.run_id}/model.npz'
    if not Path(model_path).exists():
        print(f"Error: Run ID '{args.run_id}' not found")
        print(f"Expected model at: {model_path}")
        return

    print(f"Loading model from {model_path}...\n")
    rbm = BernoulliRBM.load_model(model_path)
    print()

    # Load dataset
    print("Loading dataset for dish matching...")
    dataset = download_mm_food_100k()
    df = dataset['train'].to_pandas()
    print("Preprocessing ingredients...")
    matcher = DishMatcher(df)
    print(f"Ready to match against {len(df):,} dishes\n")

    # Run interactive session
    interactive_session(rbm, matcher, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Interactive dish recommendation with preference learning'
    )

    # Model specification
    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID to load (e.g., dwma)')

    # User preference parameters
    parser.add_argument('--latent-dim', type=int, default=50,
                       help='Dimension of user preference vector (default: 50)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Strength of preference conditioning, 0-1 (default: 0.5)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for preference updates (default: 0.1)')

    # Generation parameters
    parser.add_argument('--n-gibbs', type=int, default=1000,
                       help='Number of Gibbs sampling steps (default: 1000)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of dishes to find per recommendation (default: 3)')

    args = parser.parse_args()
    main(args)

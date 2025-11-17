"""
Main recommendation system - continuously recommends dishes and learns from feedback.

Usage:
    python main.py --run-id dwma
"""

import argparse
import numpy as np
import json
from pathlib import Path
from model.rbm import BernoulliRBM
from dataset import download_mm_food_100k
from model.ingredient_graph import parse_ingredients


class DishMatcher:
    """Match generated ingredient lists to actual dishes in the dataset."""

    def __init__(self, df):
        self.df = df
        self._preprocess_ingredients()

    def _preprocess_ingredients(self):
        """Parse all ingredients in the dataset."""
        self.df['ingredient_list'] = self.df['ingredients'].apply(parse_ingredients)
        self.df['ingredient_set'] = self.df['ingredient_list'].apply(set)

    def jaccard_similarity(self, set1, set2):
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def search_dishes(self, generated_ingredients, top_k=3):
        """Search for dishes that best match the generated ingredients."""
        generated_set = set(generated_ingredients)

        similarities = []
        for idx, row in self.df.iterrows():
            dish_ingredients = row['ingredient_set']
            score = self.jaccard_similarity(generated_set, dish_ingredients)

            similarities.append({
                'score': score,
                'dish_name': row['dish_name'],
                'ingredients': row['ingredient_list'],
                'food_type': row['food_type'],
                'cooking_method': row['cooking_method'],
                'image_url': row['image_url'],
                'nutritional_profile': row['nutritional_profile'],
            })

        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]


def display_dish(dish, ingredient_names):
    """Display dish information."""
    print(f"\n{'='*80}")
    print(f"RECOMMENDED DISH")
    print(f"{'='*80}")
    print(f"Name: {dish['dish_name']}")
    print(f"Type: {dish['food_type']}")
    if dish['cooking_method']:
        print(f"Cooking: {dish['cooking_method']}")
    print(f"Match Score: {dish['score']:.3f}")

    print(f"\nIngredients: {', '.join(dish['ingredients'])}")

    # Show matching
    generated_set = set(ingredient_names)
    dish_set = set(dish['ingredients'])
    matching = generated_set & dish_set
    if matching:
        print(f"Matching: {', '.join(sorted(matching))}")

    # Nutrition
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

    print(f"\nImage: {dish['image_url']}")
    print(f"{'='*80}")


def main(args):
    # Load model
    model_path = f'saved/{args.run_id}/model.npz'
    if not Path(model_path).exists():
        print(f"Error: Run ID '{args.run_id}' not found")
        return

    print(f"Loading model from {model_path}...")
    rbm = BernoulliRBM.load_model(model_path)
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = download_mm_food_100k()
    df = dataset['train'].to_pandas()
    print("Preprocessing ingredients...")
    matcher = DishMatcher(df)
    print(f"Ready! Dataset: {len(df):,} dishes\n")

    # Initialize user preference vector
    user_pref = rbm.init_user_preference(latent_dim=args.latent_dim)
    print(f"User preference initialized (dim={args.latent_dim})")
    print(f"Controls: 0=reject, 1=accept, q=quit")
    print(f"{'='*80}\n")

    # Stats
    iteration = 0
    accepted = 0
    rejected = 0

    # Main loop
    while True:
        iteration += 1
        print(f"[Iteration {iteration}] Stats: {accepted} accepted, {rejected} rejected")

        # Generate ingredients based on preferences
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

        print(f"Generated ingredients: {', '.join(ingredient_names)}")

        # Find matching dishes
        matches = matcher.search_dishes(ingredient_names, top_k=1)

        if not matches or matches[0]['score'] < 0.01:
            print("No good match found, generating again...\n")
            continue

        # Show top dish
        top_dish = matches[0]
        display_dish(top_dish, ingredient_names)

        # Get feedback
        while True:
            try:
                feedback_input = input("\nYour feedback (0=reject, 1=accept, q=quit): ").strip().lower()

                if feedback_input in ['q', 'quit']:
                    print(f"\n{'='*80}")
                    print("SESSION SUMMARY")
                    print(f"{'='*80}")
                    print(f"Total iterations: {iteration}")
                    print(f"Accepted: {accepted}")
                    print(f"Rejected: {rejected}")
                    print(f"Final preference norm: {np.linalg.norm(user_pref):.3f}")
                    print("\nGoodbye!")
                    return

                elif feedback_input == '1':
                    print(" Accepted")
                    user_pref = rbm.update_user_preference(
                        user_pref,
                        ingredient_names,
                        'accept',
                        learning_rate=args.learning_rate
                    )
                    accepted += 1
                    print(f"Preference norm: {np.linalg.norm(user_pref):.3f}\n")
                    break

                elif feedback_input == '0':
                    print(" Rejected")
                    user_pref = rbm.update_user_preference(
                        user_pref,
                        ingredient_names,
                        'reject',
                        learning_rate=args.learning_rate
                    )
                    rejected += 1
                    print(f"Preference norm: {np.linalg.norm(user_pref):.3f}\n")
                    break

                else:
                    print("Invalid input. Enter 0 (reject), 1 (accept), or q (quit)")

            except (EOFError, KeyboardInterrupt):
                print("\n\nSession interrupted.")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuous dish recommendation system')

    parser.add_argument('--run-id', type=str, required=True,
                       help='Run ID to load (e.g., dwma)')
    parser.add_argument('--latent-dim', type=int, default=50,
                       help='Dimension of user preference vector (default: 50)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Strength of preference conditioning (default: 0.5)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for preference updates (default: 0.1)')
    parser.add_argument('--n-gibbs', type=int, default=1000,
                       help='Gibbs sampling steps (default: 1000)')

    args = parser.parse_args()
    main(args)

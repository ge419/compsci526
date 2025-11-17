import pandas as pd
import numpy as np
import json
from dataset import download_mm_food_100k
from model.ingredient_graph import IngredientGraph


def parse_ingredients(ingredients):
    """
    Parse ingredients from various formats into a list.
    Handles: list, string representation of list, comma-separated string.
    """
    if ingredients is None or (isinstance(ingredients, float) and np.isnan(ingredients)):
        return []

    # Already a list
    if isinstance(ingredients, list):
        return ingredients

    # String representation of a list
    if isinstance(ingredients, str):
        # Try to parse as JSON first (handles '["item1", "item2"]')
        if ingredients.strip().startswith('['):
            try:
                parsed = json.loads(ingredients)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: split by comma
        return [ing.strip() for ing in ingredients.split(',') if ing.strip()]

    return []


# Simple test to verify IngredientGraph works
def test_ingredient_graph():
    print("\n" + "="*50)
    print("Testing IngredientGraph with simple data...")
    print("="*50)

    # Create simple test data
    test_data = pd.DataFrame({
        'dish_name': ['Pasta', 'Salad', 'Pizza', 'Soup'],
        'ingredients': [
            ['tomato', 'pasta', 'garlic', 'olive oil'],
            ['lettuce', 'tomato', 'cucumber', 'olive oil'],
            ['tomato', 'cheese', 'dough', 'olive oil'],
            ['tomato', 'onion', 'garlic', 'carrot']
        ]
    })

    # Extract ingredients
    all_ingredients = set()
    for ingredients in test_data['ingredients']:
        all_ingredients.update(ingredients)

    print(f"Ingredients: {sorted(all_ingredients)}")

    # Build graph
    graph = IngredientGraph(all_ingredients)
    graph.build_from_dataframe(test_data, ingredients_col='ingredients')

    print(f"\n{graph}")
    print(f"Graph shape: A={graph.A.shape}, w={graph.w.shape}")

    # Verify node weights are all 1
    print(f"\nNode weights (w): All set to 1? {np.all(graph.w == 1.0)}")
    print(f"Sample node weights: {graph.w[:5]}")

    # Test edge weights (co-occurrence)
    print(f"\nEdge weights (based on co-occurrence frequency):")
    print(f"Tomato appears with garlic: {graph.get_cooccurrence('tomato', 'garlic')} times")
    print(f"Tomato appears with olive oil: {graph.get_cooccurrence('tomato', 'olive oil')} times")

    # Test top pairings
    print(f"\nTop pairings for 'tomato' (higher weight = more frequent pairing):")
    for ing, weight in graph.get_top_pairings('tomato', top_k=3):
        print(f"  - {ing}: {weight:.0f} co-occurrences")

    print("\n✓ Test passed!")
    return graph

# Run test
test_graph = test_ingredient_graph()

# Now process the full dataset
print("\n" + "="*50)
print("Processing full dataset...")
print("="*50)

dataset = download_mm_food_100k()
df = dataset['train'].to_pandas()
df = df[['dish_name', 'ingredients', 'portion_size', 'nutritional_profile', 'cooking_method']]

# Extract all unique ingredients
all_ingredients = set()
for ingredients in df['ingredients']:
    ingredient_list = parse_ingredients(ingredients)
    all_ingredients.update(ingredient_list)

print(f"\nTotal unique ingredients: {len(all_ingredients)}")

# Build the full graph
graph = IngredientGraph(all_ingredients)
graph.build_from_dataframe(df, ingredients_col='ingredients')

print(f"\n{graph}")
print(f"Total edges (co-occurrences): {(graph.A > 0).sum() // 2}")

# Save the graph
graph_path = "dataset/ingredient_graph.npz"
graph.save(graph_path)
print(f"\nGraph saved to {graph_path}")



import pandas as pd
from dataset import download_mm_food_100k

# Download the dataset
dataset = download_mm_food_100k()

# Convert to pandas DataFrame
# If the dataset has splits (train, test, validation), access them like this:
print("Available splits:", dataset.keys())

df = dataset['train'].to_pandas() 

df = df[['dish_name', 'ingredients', 'portion_size', 'nutritional_profile', 'cooking_method']]
print(df.head())

# Extract all unique ingredients into a set
all_ingredients = set()

for ingredients in df['ingredients']:
    # Handle different formats: list, string, etc.
    if isinstance(ingredients, list):
        # If ingredients is already a list
        all_ingredients.update(ingredients)
    elif isinstance(ingredients, str):
        # If ingredients is a string, split by comma or other delimiter
        ingredient_list = [ing.strip() for ing in ingredients.split(',')]
        all_ingredients.update(ingredient_list)

print(f"\nTotal unique ingredients: {len(all_ingredients)}")
print(f"\nFirst 20 ingredients (sorted):")
print(sorted(list(all_ingredients))[:20])


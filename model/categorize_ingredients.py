"""
Script to automatically categorize ingredients into dietary restriction categories.

Categories:
- vegan: Plant-based only (no meat, dairy, eggs, honey, gelatin, etc.)
- vegetarian: No meat (allows dairy and eggs)
- none: All ingredients allowed
"""

import numpy as np
import json

# Load ingredients from model
data = np.load('saved/gtcz/model.npz', allow_pickle=True)
ingredients = [ing.lower() for ing in list(data['ingredients'])]

# Define keywords for non-vegan/vegetarian ingredients
MEAT_KEYWORDS = [
    'beef', 'pork', 'chicken', 'turkey', 'duck', 'lamb', 'mutton', 'goat',
    'bacon', 'ham', 'sausage', 'salami', 'prosciutto', 'pepperoni',
    'fish', 'salmon', 'tuna', 'cod', 'shrimp', 'prawn', 'crab', 'lobster',
    'oyster', 'clam', 'mussel', 'scallop', 'squid', 'octopus', 'anchov',
    'meat', 'steak', 'patty', 'wings', 'thigh', 'breast', 'drumstick',
    'seafood', 'sardine', 'mackerel', 'tilapia', 'catfish', 'trout',
    'venison', 'veal', 'bison', 'rabbit', 'quail', 'liver', 'kidney',
    'chorizo', 'bologna', 'pastrami', 'carne', 'pollo', 'pescado'
]

DAIRY_KEYWORDS = [
    'milk', 'cream', 'butter', 'cheese', 'yogurt', 'yoghurt', 'ghee',
    'whey', 'casein', 'lactose', 'curd', 'paneer', 'ricotta', 'mozzarella',
    'parmesan', 'cheddar', 'feta', 'brie', 'gouda', 'dairy', 'ice cream',
    'sour cream', 'buttermilk', 'condensed milk', 'evaporated milk'
]

EGG_KEYWORDS = [
    'egg', 'eggs', 'yolk', 'albumen', 'mayonnaise', 'mayo'
]

ANIMAL_PRODUCT_KEYWORDS = [
    'honey', 'gelatin', 'gelatine', 'lard', 'tallow', 'bone', 'marrow'
]

def contains_keyword(ingredient: str, keywords: list) -> bool:
    """Check if ingredient contains any of the keywords."""
    # Special cases - these are vegan despite containing keywords
    vegan_exceptions = {
        'coconut milk', 'almond milk', 'soy milk', 'oat milk', 'rice milk',
        'eggplant'
    }

    # If this is a vegan exception, don't flag it
    if ingredient in vegan_exceptions:
        return False

    return any(keyword in ingredient for keyword in keywords)

# Categorize each ingredient
categorization = {}

for ing in ingredients:
    # Check if ingredient contains meat
    has_meat = contains_keyword(ing, MEAT_KEYWORDS)
    has_dairy = contains_keyword(ing, DAIRY_KEYWORDS)
    has_egg = contains_keyword(ing, EGG_KEYWORDS)
    has_animal_product = contains_keyword(ing, ANIMAL_PRODUCT_KEYWORDS)

    # Determine dietary restrictions that EXCLUDE this ingredient
    restricted_for = []

    if has_meat:
        restricted_for.extend(['vegan', 'vegetarian'])
    if has_dairy or has_egg or has_animal_product:
        restricted_for.append('vegan')

    categorization[ing] = {
        'restricted_for': restricted_for,
        'is_meat': has_meat,
        'is_dairy': has_dairy,
        'is_egg': has_egg,
        'is_animal_product': has_animal_product
    }

# Save categorization
with open('dietary_restrictions.json', 'w') as f:
    json.dump(categorization, f, indent=2)

# Print statistics
print(f"Total ingredients: {len(ingredients)}")
print(f"\nRestriction statistics:")

vegan_excluded = sum(1 for ing, cat in categorization.items() if 'vegan' in cat['restricted_for'])
vegetarian_excluded = sum(1 for ing, cat in categorization.items() if 'vegetarian' in cat['restricted_for'])
vegan_allowed = len(ingredients) - vegan_excluded
vegetarian_allowed = len(ingredients) - vegetarian_excluded

print(f"  Vegan: {vegan_allowed} allowed, {vegan_excluded} excluded")
print(f"  Vegetarian: {vegetarian_allowed} allowed, {vegetarian_excluded} excluded")
print(f"  None: {len(ingredients)} allowed, 0 excluded")

# Print some examples
print("\n\nExamples of categorized ingredients:")
print("\nMeat (excluded for vegan & vegetarian):")
meat_items = [ing for ing, cat in categorization.items() if cat['is_meat']][:10]
for item in meat_items:
    print(f"  - {item}")

print("\nDairy (excluded for vegan only):")
dairy_items = [ing for ing, cat in categorization.items() if cat['is_dairy']][:10]
for item in dairy_items:
    print(f"  - {item}")

print("\nEggs (excluded for vegan only):")
egg_items = [ing for ing, cat in categorization.items() if cat['is_egg']][:10]
for item in egg_items:
    print(f"  - {item}")

print("\nVegan-friendly (first 20):")
vegan_items = [ing for ing, cat in categorization.items() if not cat['restricted_for']][:20]
for item in vegan_items:
    print(f"  - {item}")

print(f"\nCategorization saved to dietary_restrictions.json")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter
from dataset import download_mm_food_100k
from model.ingredient_graph import IngredientGraph

# Create fig directory if it doesn't exist
os.makedirs('fig', exist_ok=True)


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


def load_data():
    """Load and prepare the dataset."""
    print("Loading dataset...")
    dataset = download_mm_food_100k()
    df = dataset['train'].to_pandas()
    return df


def basic_statistics(df):
    """Display basic statistics about the dataset."""
    print("\n" + "="*60)
    print("BASIC DATASET STATISTICS")
    print("="*60)

    print(f"\nTotal number of dishes: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"\nColumn names: {list(df.columns)}")

    print(f"\nDataset shape: {df.shape}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    return df


def ingredient_analysis(df):
    """Analyze ingredients in the dataset."""
    print("\n" + "="*60)
    print("INGREDIENT ANALYSIS")
    print("="*60)

    # Extract all ingredients
    all_ingredients = []
    ingredient_counts_per_dish = []

    for ingredients in df['ingredients']:
        ingredient_list = parse_ingredients(ingredients)
        all_ingredients.extend(ingredient_list)
        ingredient_counts_per_dish.append(len(ingredient_list))

    # Basic stats
    unique_ingredients = set(all_ingredients)
    print(f"\nTotal unique ingredients: {len(unique_ingredients)}")
    print(f"Total ingredient mentions: {len(all_ingredients)}")

    # Ingredient count per dish
    print(f"\nIngredients per dish:")
    print(f"  Mean: {np.mean(ingredient_counts_per_dish):.2f}")
    print(f"  Median: {np.median(ingredient_counts_per_dish):.2f}")
    print(f"  Min: {min(ingredient_counts_per_dish)}")
    print(f"  Max: {max(ingredient_counts_per_dish)}")
    print(f"  Std: {np.std(ingredient_counts_per_dish):.2f}")

    # Most common ingredients
    ingredient_counter = Counter(all_ingredients)
    print(f"\nTop 20 most common ingredients:")
    for ing, count in ingredient_counter.most_common(20):
        print(f"  {ing}: {count} dishes ({count/len(df)*100:.2f}%)")

    return all_ingredients, unique_ingredients, ingredient_counts_per_dish


def cooking_method_analysis(df):
    """Analyze cooking methods in the dataset."""
    print("\n" + "="*60)
    print("COOKING METHOD ANALYSIS")
    print("="*60)

    if 'cooking_method' not in df.columns:
        print("No cooking_method column found")
        return

    # Count cooking methods
    cooking_methods = df['cooking_method'].value_counts()
    print(f"\nTotal unique cooking methods: {len(cooking_methods)}")
    print(f"\nTop 15 cooking methods:")
    for method, count in cooking_methods.head(15).items():
        print(f"  {method}: {count} dishes ({count/len(df)*100:.2f}%)")

    return cooking_methods


def portion_size_analysis(df):
    """Analyze portion sizes in the dataset."""
    print("\n" + "="*60)
    print("PORTION SIZE ANALYSIS")
    print("="*60)

    if 'portion_size' not in df.columns:
        print("No portion_size column found")
        return

    portion_sizes = df['portion_size'].value_counts()
    print(f"\nTotal unique portion sizes: {len(portion_sizes)}")
    print(f"\nTop 10 portion sizes:")
    for size, count in portion_sizes.head(10).items():
        print(f"  {size}: {count} dishes ({count/len(df)*100:.2f}%)")

    return portion_sizes


def nutritional_analysis(df):
    """Analyze nutritional profiles and macronutrients in the dataset."""
    print("\n" + "="*60)
    print("NUTRITIONAL PROFILE & MACRONUTRIENT ANALYSIS")
    print("="*60)

    if 'nutritional_profile' not in df.columns:
        print("No nutritional_profile column found")
        return None

    print(f"\nSample nutritional profiles:")
    print(df['nutritional_profile'].head(3))
    print(f"\nNutritional profile data type: {type(df['nutritional_profile'].iloc[0])}")

    # Parse nutritional data
    nutritional_data = {
        'calories': [],
        'protein': [],
        'carbohydrates': [],
        'fat': [],
        'fiber': [],
        'sugar': [],
        'sodium': []
    }

    for profile in df['nutritional_profile']:
        if profile is None or (isinstance(profile, float) and np.isnan(profile)):
            continue

        # Parse JSON string to dict
        profile_dict = None
        if isinstance(profile, str):
            try:
                profile_dict = json.loads(profile)
            except json.JSONDecodeError:
                continue
        elif isinstance(profile, dict):
            profile_dict = profile
        else:
            continue

        # Extract nutritional values (handle various key formats)
        if profile_dict:
            nutritional_data['calories'].append(
                profile_dict.get('calories', profile_dict.get('calories_kcal', np.nan))
            )
            nutritional_data['protein'].append(
                profile_dict.get('protein', profile_dict.get('protein_g', np.nan))
            )
            nutritional_data['carbohydrates'].append(
                profile_dict.get('carbohydrates', profile_dict.get('carbohydrate_g', profile_dict.get('carbs', np.nan)))
            )
            nutritional_data['fat'].append(
                profile_dict.get('fat', profile_dict.get('fat_g', np.nan))
            )
            nutritional_data['fiber'].append(
                profile_dict.get('fiber', profile_dict.get('fiber_g', np.nan))
            )
            nutritional_data['sugar'].append(
                profile_dict.get('sugar', profile_dict.get('sugar_g', np.nan))
            )
            nutritional_data['sodium'].append(
                profile_dict.get('sodium', profile_dict.get('sodium_mg', np.nan))
            )

    # Create DataFrame for easier analysis
    nutrition_df = pd.DataFrame(nutritional_data)

    # Remove NaN values for statistics
    print(f"\n{'Macronutrient':<15} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12} {'Std':<12}")
    print("-" * 75)

    for nutrient in ['calories', 'protein', 'carbohydrates', 'fat', 'fiber', 'sugar', 'sodium']:
        values = [v for v in nutritional_data[nutrient] if not (isinstance(v, float) and np.isnan(v))]
        if values:
            values = np.array(values, dtype=float)
            print(f"{nutrient:<15} {np.mean(values):<12.2f} {np.median(values):<12.2f} "
                  f"{np.min(values):<12.2f} {np.max(values):<12.2f} {np.std(values):<12.2f}")
        else:
            print(f"{nutrient:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    # Macronutrient ratios
    print("\n" + "="*60)
    print("MACRONUTRIENT RATIOS")
    print("="*60)

    protein_vals = [v for v in nutritional_data['protein'] if not (isinstance(v, float) and np.isnan(v))]
    carb_vals = [v for v in nutritional_data['carbohydrates'] if not (isinstance(v, float) and np.isnan(v))]
    fat_vals = [v for v in nutritional_data['fat'] if not (isinstance(v, float) and np.isnan(v))]

    if protein_vals and carb_vals and fat_vals:
        total_protein = np.sum(protein_vals)
        total_carbs = np.sum(carb_vals)
        total_fat = np.sum(fat_vals)
        total = total_protein + total_carbs + total_fat

        print(f"\nAverage macronutrient distribution:")
        print(f"  Protein: {total_protein/total*100:.1f}%")
        print(f"  Carbohydrates: {total_carbs/total*100:.1f}%")
        print(f"  Fat: {total_fat/total*100:.1f}%")

    return nutrition_df


def graph_analysis(df, all_ingredients):
    """Analyze the ingredient graph structure."""
    print("\n" + "="*60)
    print("INGREDIENT GRAPH ANALYSIS")
    print("="*60)

    # Build graph
    print("\nBuilding ingredient graph...")
    unique_ingredients = set(all_ingredients)
    graph = IngredientGraph(unique_ingredients)
    graph.build_from_dataframe(df, ingredients_col='ingredients')

    print(f"\n{graph}")
    print(f"Graph density: {np.count_nonzero(graph.A) / (graph.N * graph.N):.4f}")
    print(f"Total edges (non-zero co-occurrences): {np.count_nonzero(graph.A) // 2}")

    # Average degree
    degrees = np.sum(graph.A > 0, axis=1)
    print(f"\nNode degree statistics:")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    print(f"  Median degree: {np.median(degrees):.2f}")
    print(f"  Max degree: {np.max(degrees)}")
    print(f"  Min degree: {np.min(degrees)}")

    # Most connected ingredients
    print(f"\nTop 10 most connected ingredients (by degree):")
    top_degree_indices = np.argsort(degrees)[::-1][:10]
    for idx in top_degree_indices:
        ing = graph.get_ingredient_name(idx)
        print(f"  {ing}: {int(degrees[idx])} connections")

    # Strongest pairings
    print(f"\nTop 10 strongest ingredient pairings (by co-occurrence):")
    edges = []
    for i in range(graph.N):
        for j in range(i+1, graph.N):
            if graph.A[i, j] > 0:
                edges.append((graph.get_ingredient_name(i),
                            graph.get_ingredient_name(j),
                            graph.A[i, j]))
    edges.sort(key=lambda x: x[2], reverse=True)
    for ing1, ing2, weight in edges[:10]:
        print(f"  {ing1} + {ing2}: {int(weight)} co-occurrences")

    return graph


def visualize_distributions(df, ingredient_counts_per_dish, cooking_methods, portion_sizes):
    """Create visualizations for the data distributions."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Ingredient count distribution
    axes[0, 0].hist(ingredient_counts_per_dish, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Number of Ingredients')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Ingredients per Dish')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Top cooking methods
    top_methods = cooking_methods.head(10)
    axes[0, 1].barh(range(len(top_methods)), top_methods.values)
    axes[0, 1].set_yticks(range(len(top_methods)))
    axes[0, 1].set_yticklabels(top_methods.index)
    axes[0, 1].set_xlabel('Count')
    axes[0, 1].set_title('Top 10 Cooking Methods')
    axes[0, 1].invert_yaxis()

    # 3. Top portion sizes
    top_portions = portion_sizes.head(10)
    axes[1, 0].barh(range(len(top_portions)), top_portions.values)
    axes[1, 0].set_yticks(range(len(top_portions)))
    axes[1, 0].set_yticklabels(top_portions.index)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Top 10 Portion Sizes')
    axes[1, 0].invert_yaxis()

    # 4. Ingredient count box plot
    axes[1, 1].boxplot(ingredient_counts_per_dish, vert=True)
    axes[1, 1].set_ylabel('Number of Ingredients')
    axes[1, 1].set_title('Ingredient Count Distribution (Box Plot)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig/eda_visualizations.png', dpi=300, bbox_inches='tight')
    print("\nVisualizations saved to: fig/eda_visualizations.png")

    return fig


def visualize_macronutrients(nutrition_df):
    """Create visualizations for macronutrient distributions."""
    print("\n" + "="*60)
    print("GENERATING MACRONUTRIENT VISUALIZATIONS")
    print("="*60)

    if nutrition_df is None or nutrition_df.empty:
        print("No nutritional data available for visualization")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Calories distribution
    calories = nutrition_df['calories'].dropna()
    if len(calories) > 0:
        axes[0, 0].hist(calories, bins=50, edgecolor='black', color='orange')
        axes[0, 0].set_xlabel('Calories')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Calorie Distribution')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Protein distribution
    protein = nutrition_df['protein'].dropna()
    if len(protein) > 0:
        axes[0, 1].hist(protein, bins=50, edgecolor='black', color='red')
        axes[0, 1].set_xlabel('Protein (g)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Protein Distribution')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Carbohydrates distribution
    carbs = nutrition_df['carbohydrates'].dropna()
    if len(carbs) > 0:
        axes[0, 2].hist(carbs, bins=50, edgecolor='black', color='blue')
        axes[0, 2].set_xlabel('Carbohydrates (g)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Carbohydrate Distribution')
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Fat distribution
    fat = nutrition_df['fat'].dropna()
    if len(fat) > 0:
        axes[1, 0].hist(fat, bins=50, edgecolor='black', color='yellow')
        axes[1, 0].set_xlabel('Fat (g)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Fat Distribution')
        axes[1, 0].grid(True, alpha=0.3)

    # 5. Macronutrient comparison (box plots)
    macro_data = []
    macro_labels = []
    for macro, color in [('protein', 'red'), ('carbohydrates', 'blue'), ('fat', 'yellow')]:
        values = nutrition_df[macro].dropna()
        if len(values) > 0:
            macro_data.append(values)
            macro_labels.append(macro.capitalize())

    if macro_data:
        bp = axes[1, 1].boxplot(macro_data, labels=macro_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'blue', 'yellow']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1, 1].set_ylabel('Amount (g)')
        axes[1, 1].set_title('Macronutrient Comparison')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Macronutrient ratios (pie chart)
    if len(protein) > 0 and len(carbs) > 0 and len(fat) > 0:
        avg_protein = protein.mean()
        avg_carbs = carbs.mean()
        avg_fat = fat.mean()

        sizes = [avg_protein, avg_carbs, avg_fat]
        labels = ['Protein', 'Carbohydrates', 'Fat']
        colors = ['red', 'blue', 'yellow']
        explode = (0.05, 0.05, 0.05)

        axes[1, 2].pie(sizes, explode=explode, labels=labels, colors=colors,
                      autopct='%1.1f%%', shadow=True, startangle=90)
        axes[1, 2].set_title('Average Macronutrient Ratio')

    plt.tight_layout()
    plt.savefig('fig/macronutrient_analysis.png', dpi=300, bbox_inches='tight')
    print("\nMacronutrient visualizations saved to: fig/macronutrient_analysis.png")

    return fig


def visualize_ingredient_network(graph, top_n=30):
    """Visualize the top N most connected ingredients as a network."""
    print("\n" + "="*60)
    print("GENERATING NETWORK VISUALIZATION")
    print("="*60)

    # Get top N most connected ingredients
    degrees = np.sum(graph.A > 0, axis=1)
    top_indices = np.argsort(degrees)[::-1][:top_n]

    # Create subgraph
    subgraph_A = graph.A[np.ix_(top_indices, top_indices)]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    ingredient_names = [graph.get_ingredient_name(i) for i in top_indices]

    sns.heatmap(subgraph_A,
                xticklabels=ingredient_names,
                yticklabels=ingredient_names,
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'label': 'Co-occurrence Count'})

    ax.set_title(f'Co-occurrence Heatmap: Top {top_n} Most Connected Ingredients')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('fig/ingredient_network.png', dpi=300, bbox_inches='tight')
    print("\nNetwork visualization saved to: fig/ingredient_network.png")

    return fig


if __name__ == "__main__":
    # Load data
    df = load_data()

    # Run all analyses
    basic_statistics(df)

    all_ingredients, unique_ingredients, ingredient_counts = ingredient_analysis(df)

    cooking_methods = cooking_method_analysis(df)

    portion_sizes = portion_size_analysis(df)

    nutrition_df = nutritional_analysis(df)

    graph = graph_analysis(df, all_ingredients)

    # Generate visualizations
    if cooking_methods is not None and portion_sizes is not None:
        visualize_distributions(df, ingredient_counts, cooking_methods, portion_sizes)

    if nutrition_df is not None:
        visualize_macronutrients(nutrition_df)

    visualize_ingredient_network(graph, top_n=30)

    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - fig/eda_visualizations.png")
    print("  - fig/macronutrient_analysis.png")
    print("  - fig/ingredient_network.png")

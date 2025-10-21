import numpy as np
import pandas as pd
import json
from typing import List, Set, Dict


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


class IngredientGraph:
    """
    Undirected graph structure for ingredients.

    Attributes:
        N (int): Number of unique ingredients
        ingredients (list): List of all unique ingredients
        ingredient_to_idx (dict): Mapping from ingredient name to index
        idx_to_ingredient (dict): Mapping from index to ingredient name
        A (np.ndarray): N x N adjacency matrix storing edge weights (co-occurrence)
        w (np.ndarray): N-dimensional weight vector for node weights
    """

    def __init__(self, ingredients: Set[str]):
        """
        Initialize the ingredient graph.

        Args:
            ingredients: Set of unique ingredient names
        """
        self.ingredients = sorted(list(ingredients))
        self.N = len(self.ingredients)

        # Create mappings between ingredients and indices
        self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(self.ingredients)}
        self.idx_to_ingredient = {idx: ing for idx, ing in enumerate(self.ingredients)}

        # Initialize adjacency matrix A (N x N) - edge weights
        self.A = np.zeros((self.N, self.N), dtype=np.float32)

        # Initialize node weights w (N,) - can be set based on popularity or other metrics
        self.w = np.zeros(self.N, dtype=np.float32)

    def build_from_dataframe(self, df: pd.DataFrame, ingredients_col: str = 'ingredients'):
        """
        Build the graph by calculating co-occurrence from a dataframe.

        Edge weights (A matrix): Set based on co-occurrence frequency. If two ingredients
        appear together more frequently, their edge weight will be higher.

        Node weights (w vector): All set to 1.

        Args:
            df: DataFrame containing dishes and their ingredients
            ingredients_col: Name of the column containing ingredients
        """
        for ingredients in df[ingredients_col]:
            # Parse ingredients using helper function
            ingredient_list = parse_ingredients(ingredients)

            if not ingredient_list:
                continue

            # Get indices for these ingredients
            indices = []
            for ing in ingredient_list:
                if ing in self.ingredient_to_idx:
                    idx = self.ingredient_to_idx[ing]
                    indices.append(idx)

            # Update co-occurrence matrix (symmetric)
            # Each time two ingredients appear together in a dish, increment their edge weight
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    self.A[idx_i, idx_j] += 1
                    self.A[idx_j, idx_i] += 1  # Symmetric for undirected graph

        # Set all node weights to 1
        self.w = np.ones(self.N, dtype=np.float32)

        return self

    def normalize_edge_weights(self, method: str = 'count'):
        """
        Normalize the edge weights in the adjacency matrix.

        Args:
            method: Normalization method ('count', 'probability', 'log')
        """
        if method == 'probability':
            # Normalize by total co-occurrences
            total = np.sum(self.A)
            if total > 0:
                self.A = self.A / total
        elif method == 'log':
            # Log normalization (add 1 to avoid log(0))
            self.A = np.log1p(self.A)
        # 'count' keeps raw counts

        return self

    def get_ingredient_idx(self, ingredient: str) -> int:
        """Get index for an ingredient."""
        return self.ingredient_to_idx.get(ingredient, -1)

    def get_ingredient_name(self, idx: int) -> str:
        """Get ingredient name from index."""
        return self.idx_to_ingredient.get(idx, "")

    def get_cooccurrence(self, ing1: str, ing2: str) -> float:
        """Get co-occurrence weight between two ingredients."""
        idx1 = self.get_ingredient_idx(ing1)
        idx2 = self.get_ingredient_idx(ing2)

        if idx1 == -1 or idx2 == -1:
            return 0.0

        return self.A[idx1, idx2]

    def get_top_pairings(self, ingredient: str, top_k: int = 10) -> List[tuple]:
        """
        Get top k ingredients that pair well with the given ingredient.

        Args:
            ingredient: Ingredient name
            top_k: Number of top pairings to return

        Returns:
            List of (ingredient_name, weight) tuples
        """
        idx = self.get_ingredient_idx(ingredient)
        if idx == -1:
            return []

        # Get all edge weights for this ingredient
        weights = self.A[idx, :]

        # Get top k indices (excluding self)
        top_indices = np.argsort(weights)[::-1][:top_k]

        # Convert to ingredient names and weights
        pairings = [(self.get_ingredient_name(i), weights[i])
                    for i in top_indices if i != idx]

        return pairings

    def save(self, filepath: str):
        """Save the graph to a file."""
        np.savez(filepath,
                 A=self.A,
                 w=self.w,
                 ingredients=self.ingredients)

    def load(self, filepath: str):
        """Load the graph from a file."""
        data = np.load(filepath, allow_pickle=True)
        self.A = data['A']
        self.w = data['w']
        self.ingredients = list(data['ingredients'])
        self.N = len(self.ingredients)
        self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(self.ingredients)}
        self.idx_to_ingredient = {idx: ing for idx, ing in enumerate(self.ingredients)}
        return self

    def __repr__(self):
        return f"IngredientGraph(N={self.N}, edges={np.count_nonzero(self.A)//2})"

import os
from datasets import load_dataset


# Define the dataset cache directory
DATASET_DIR = os.path.join(os.path.dirname(__file__))


def download_mm_food_100k(streaming=False, cache_dir=None):
    """
    Download the MM-Food-100K dataset from Hugging Face.

    Args:
        streaming (bool): If True, stream the dataset instead of downloading entirely.
                         Useful for large datasets.
        cache_dir (str): Directory to cache the dataset. Defaults to ./dataset/

    Returns:
        dataset: The loaded dataset object
    """
    if cache_dir is None:
        cache_dir = DATASET_DIR

    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset(
        "Codatta/MM-Food-100K",
        streaming=streaming,
        cache_dir=cache_dir
    )
    return dataset


# You can also expose the dataset directly
# dataset = load_dataset("Codatta/MM-Food-100K", cache_dir=DATASET_DIR)

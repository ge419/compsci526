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


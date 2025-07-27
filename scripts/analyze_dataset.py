import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen

# Load the dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/DATASET-balanced-JcqFJYhgnWK5P8zrmIuuMwyj9BIpH9.csv"

print("Loading dataset...")
df = pd.read_csv(url)

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:")
print(df['LABEL'].value_counts())

print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset statistics:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum().sum())

# Check data types
print(f"\nData types:")
print(df.dtypes)

from data_loader import load_dataset
import numpy as np

df = load_dataset("wheat")
y = np.round(df["y"].values, 2)
print("Array y (synthetic values):")
print(y.tolist())
print("\nFirst 12:", y[:12].tolist())
print("Last 12:", y[-12:].tolist())
print("Min:", y.min(), "Max:", y.max(), "Mean:", round(y.mean(), 2))

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading local embedding model (one-time download, ~90MB)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

pairs = pd.read_csv("data/amazon_pairs_classified.csv")

print(f"Embedding {len(pairs)} customer messages...")
embeddings = model.encode(pairs["customer_message"].tolist(), show_progress_bar=True)

np.save("data/pair_embeddings.npy", embeddings)
pairs.to_csv("data/amazon_pairs_final.csv", index=False)
print(f"\nSaved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
print("Saved to data/pair_embeddings.npy and data/amazon_pairs_final.csv")
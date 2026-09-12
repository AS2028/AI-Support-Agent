import pandas as pd
import time
from tqdm import tqdm
from classify_intent import classify_batch

pairs = pd.read_csv("data/amazon_pairs.csv")

SAMPLE_SIZE = 300
BATCH_SIZE = 15

pairs = pairs.sample(n=min(SAMPLE_SIZE, len(pairs)), random_state=42).reset_index(drop=True)
messages = pairs["customer_message"].tolist()
chunks = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]

print(f"Classifying {len(messages)} messages in {len(chunks)} batches...")

all_labels = []
for chunk in tqdm(chunks):
    labels = classify_batch(chunk)
    all_labels.extend(labels)
    time.sleep(1)

pairs["predicted_intent"] = all_labels

print("\nIntent distribution:")
print(pairs["predicted_intent"].value_counts())

pairs.to_csv("data/amazon_pairs_classified.csv", index=False)
print("\nSaved to data/amazon_pairs_classified.csv")
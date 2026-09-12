import pandas as pd

# Load the full dataset
df = pd.read_csv("data/twcs.csv")

# Step 1: Isolate AmazonHelp's replies (inbound=False means it's the brand replying, not the customer)
amazon_replies = df[df["author_id"] == "AmazonHelp"]
print(f"Total AmazonHelp replies: {len(amazon_replies)}")

# Step 2: Filter to (likely) English text only
# Heuristic: if less than 10% of characters are non-ASCII, treat as English
def is_likely_english(text):
    if not isinstance(text, str):
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(len(text), 1)) < 0.1

amazon_replies_en = amazon_replies[amazon_replies["text"].apply(is_likely_english)]
print(f"English-only AmazonHelp replies: {len(amazon_replies_en)}")

# Step 3: Preview a handful of real replies to sanity-check
print("\nSample replies:")
for text in amazon_replies_en["text"].head(5):
    print(f"- {text[:150]}")
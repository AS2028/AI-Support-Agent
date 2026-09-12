import pandas as pd
from langdetect import detect, LangDetectException
from tqdm import tqdm

tqdm.pandas()  # enables .progress_apply() with a visible progress bar

df = pd.read_csv("data/twcs.csv")
tweets_by_id = df.set_index("tweet_id")
amazon_replies = df[df["author_id"] == "AmazonHelp"].copy()

def get_customer_message(row):
    parent_id = row["in_response_to_tweet_id"]
    if pd.isna(parent_id):
        return None
    parent_id = int(parent_id)
    if parent_id in tweets_by_id.index:
        return tweets_by_id.loc[parent_id, "text"]
    return None

amazon_replies["customer_message"] = amazon_replies.apply(get_customer_message, axis=1)
pairs = amazon_replies.dropna(subset=["customer_message"])
pairs = pairs[["customer_message", "text"]].rename(columns={"text": "amazon_reply"})

# Subsample BEFORE the slow step — explicitly allowed by the assignment
SAMPLE_SIZE = 8000
pairs = pairs.sample(n=min(SAMPLE_SIZE, len(pairs)), random_state=42)
print(f"Working with {len(pairs)} pairs (out of {len(amazon_replies)} total)")

def is_english(text):
    if not isinstance(text, str) or len(text.strip()) < 3:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

print("Detecting language on customer messages...")
customer_mask = pairs["customer_message"].progress_apply(is_english)
print("Detecting language on Amazon replies...")
amazon_mask = pairs["amazon_reply"].progress_apply(is_english)

pairs = pairs[customer_mask & amazon_mask]

print(f"\nTotal valid English pairs: {len(pairs)}")
for _, row in pairs.head(5).iterrows():
    print(f"CUSTOMER: {row['customer_message'][:150]}")
    print(f"AMAZON:   {row['amazon_reply'][:150]}")
    print("---")

pairs.to_csv("data/amazon_pairs.csv", index=False)
print("Saved to data/amazon_pairs.csv")
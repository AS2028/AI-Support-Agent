import pandas as pd

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

# IMPORTANT: different random_state (99) so this NEVER overlaps with the 8000-sample or 300-pair RAG corpus
golden_candidates = pairs.sample(n=250, random_state=99).reset_index(drop=True)

golden_candidates["true_intent"] = ""  # you'll fill this in by hand
golden_candidates["should_escalate"] = ""  # you'll fill this in by hand (True/False)
golden_candidates["notes"] = ""

golden_candidates.to_csv("data/golden_eval_raw.csv", index=False)
print(f"Saved {len(golden_candidates)} candidate messages to data/golden_eval_raw.csv")
print("Open this in Excel/Google Sheets and fill in 'true_intent' and 'should_escalate' columns by hand.")
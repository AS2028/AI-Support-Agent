import pandas as pd
from classify_intent import classify_batch
from escalate import should_escalate

golden = pd.read_csv("data/golden_eval_raw.csv")

messages = golden["customer_message"].tolist()
BATCH_SIZE = 15
chunks = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]

print("Pre-filling intent predictions for review...")
all_labels = []
for chunk in chunks:
    all_labels.extend(classify_batch(chunk))

golden["true_intent"] = all_labels  # pre-filled - YOU will review/correct these
golden["should_escalate"] = [should_escalate(m, i)[0] for m, i in zip(messages, all_labels)]

golden.to_csv("data/golden_eval_prefilled.csv", index=False)
print(f"Saved {len(golden)} pre-filled rows to data/golden_eval_prefilled.csv")
print("IMPORTANT: These are DRAFT labels from the model — you must review each one and correct any that are wrong before this counts as a real golden set.")
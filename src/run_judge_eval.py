import pandas as pd
from tqdm import tqdm
from generate_reply import generate_grounded_reply
from llm_judge import judge_reply

golden = pd.read_csv("data/golden_eval_raw.csv", encoding="latin1")
golden = golden[golden["true_intent"].notna() & (golden["true_intent"] != "") & (golden["true_intent"] != "SKIP")].reset_index(drop=True)

# Judge a subset (25 examples) - enough for meaningful human-agreement check without excessive API calls
JUDGE_SAMPLE_SIZE = 25
sample = golden.sample(n=min(JUDGE_SAMPLE_SIZE, len(golden)), random_state=7).reset_index(drop=True)

results = []
print(f"Generating replies and judge scores for {len(sample)} examples...")
for _, row in tqdm(sample.iterrows(), total=len(sample)):
    msg = row["customer_message"]
    reply = generate_grounded_reply(msg)
    scores = judge_reply(msg, reply)
    results.append({
        "customer_message": msg,
        "generated_reply": reply,
        "judge_relevance": scores["relevance"],
        "judge_tone": scores["tone"],
        "judge_resolution": scores["resolution"],
        "human_relevance": "",   # YOU will fill this in
        "human_tone": "",        # YOU will fill this in
        "human_resolution": "",  # YOU will fill this in
    })

out = pd.DataFrame(results)
out.to_csv("data/judge_vs_human.csv", index=False)
print(f"\nSaved to data/judge_vs_human.csv")
print("Open this file and fill in human_relevance/human_tone/human_resolution (1-5 each) for every row,")
print("using your own judgment WITHOUT looking at the judge_* columns first (cover them or don't peek).")
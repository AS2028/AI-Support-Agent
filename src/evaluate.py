import pandas as pd
from classify_intent import classify_batch
from escalate import should_escalate
from baseline_trivial import trivial_baseline
from baseline_simple import simple_baseline

golden = pd.read_csv("data/golden_eval_raw.csv", encoding="latin1")
golden = golden[golden["true_intent"].notna() & (golden["true_intent"] != "") & (golden["true_intent"] != "SKIP")].reset_index(drop=True)
print(f"Evaluating on {len(golden)} labeled golden examples\n")

messages = golden["customer_message"].tolist()
true_intents = golden["true_intent"].tolist()
true_escalate = golden["should_escalate"].astype(str).str.upper().eq("TRUE").tolist()

print("Running my classifier...")
my_predictions = []
BATCH_SIZE = 15
for i in range(0, len(messages), BATCH_SIZE):
    chunk = messages[i:i+BATCH_SIZE]
    my_predictions.extend(classify_batch(chunk))

trivial_predictions = trivial_baseline(messages)
simple_predictions = simple_baseline(messages)

def accuracy(preds, truth):
    correct = sum(1 for p, t in zip(preds, truth) if p == t)
    return correct / len(truth)

print("\n=== INTENT CLASSIFICATION ACCURACY ===")
print(f"Trivial baseline:  {accuracy(trivial_predictions, true_intents):.1%}")
print(f"Simple baseline:   {accuracy(simple_predictions, true_intents):.1%}")
print(f"My classifier:     {accuracy(my_predictions, true_intents):.1%}")

my_escalate_preds = [should_escalate(m, intent)[0] for m, intent in zip(messages, my_predictions)]
escalate_correct = sum(1 for p, t in zip(my_escalate_preds, true_escalate) if p == t)
print(f"\n=== ESCALATION DECISION ACCURACY ===")
print(f"My escalation logic: {escalate_correct/len(true_escalate):.1%}")

golden["my_predicted_intent"] = my_predictions
golden["trivial_predicted_intent"] = trivial_predictions
golden["simple_predicted_intent"] = simple_predictions
golden["my_predicted_escalate"] = my_escalate_preds
golden["intent_correct"] = golden["my_predicted_intent"] == golden["true_intent"]
golden.to_csv("data/evaluation_results.csv", index=False)
print("\nSaved detailed results to data/evaluation_results.csv")
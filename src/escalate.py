# Intents that inherently warrant human review, regardless of confidence
SENSITIVE_INTENTS = {"Account/Security Issue", "Billing/Charge Issue"}

# Phrases signaling the customer is already frustrated/escalated on their own
FRUSTRATION_SIGNALS = [
    "no one", "still waiting", "third time", "again", "unacceptable",
    "ridiculous", "pathetic", "terrible", "worst", "furious", "lawsuit",
    "lawyer", "cancel my", "never again", "scam"
]

def should_escalate(message: str, predicted_intent: str) -> tuple:
    """Returns (should_escalate: bool, reason: str)"""
    message_lower = message.lower()

    if predicted_intent == "Other/Unclear":
        return True, "Intent could not be confidently classified"

    if predicted_intent in SENSITIVE_INTENTS:
        return True, f"Intent '{predicted_intent}' involves account/financial risk and requires human handling"

    frustration_hits = [phrase for phrase in FRUSTRATION_SIGNALS if phrase in message_lower]
    if frustration_hits:
        return True, f"Message shows signs of customer frustration/escalation (matched: {frustration_hits[0]})"

    if len(message.strip()) < 10:
        return True, "Message too short to confidently assess"

    return False, f"Routine '{predicted_intent}' case with no risk/frustration signals — safe to auto-handle"

if __name__ == "__main__":
    tests = [
        ("Where is my package? It was supposed to arrive yesterday!", "Delivery Issue"),
        ("This is the third time I've contacted you, unacceptable service!", "Delivery Issue"),
        ("I can't log into my account", "Account/Security Issue"),
        ("Do you have any deals on Kindles?", "General Product/Service Question"),
    ]
    for msg, intent in tests:
        escalate, reason = should_escalate(msg, intent)
        print(f"MESSAGE: {msg}")
        print(f"INTENT: {intent} | ESCALATE: {escalate} | REASON: {reason}")
        print("---")
import sys
from classify_intent import classify_batch
from generate_reply import generate_grounded_reply
from escalate import should_escalate

def process_message(message: str) -> dict:
    intent = classify_batch([message])[0]
    escalate, reason = should_escalate(message, intent)
    reply = None if escalate else generate_grounded_reply(message)
    return {
        "message": message,
        "intent": intent,
        "escalate": escalate,
        "escalation_reason": reason,
        "reply": reply,
    }

if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "Where is my package? It's 3 days late."
    result = process_message(message)
    print(f"MESSAGE: {result['message']}")
    print(f"INTENT: {result['intent']}")
    print(f"ESCALATE: {result['escalate']} ({result['escalation_reason']})")
    if result['reply']:
        print(f"REPLY: {result['reply']}")
    else:
        print("REPLY: [Routed to human agent, no auto-reply generated]")
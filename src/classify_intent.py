import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

INTENTS = [
    "Delivery Issue",
    "Damaged/Defective Item",
    "Refund/Return Request",
    "Order Status/Tracking Inquiry",
    "Account/Security Issue",
    "Billing/Charge Issue",
    "General Product/Service Question",
    "Other/Unclear",
]

BATCH_PROMPT = """Classify each Amazon customer support message into ONE intent.

Intents:
{intents}

Rules:
- Delivery Issue: late/missing/never arrived package
- Damaged/Defective Item: item arrived broken or stopped working
- Refund/Return Request: wants money back or to return item
- Order Status/Tracking Inquiry: neutral question about order location
- Account/Security Issue: login/hacked account/access issues
- Billing/Charge Issue: unexpected or incorrect charges
- General Product/Service Question: informational, not an issue
- Other/Unclear: doesn't fit above

Return ONLY a JSON array of exactly {n} strings (one intent per message, same order). No explanation.

Messages:
{messages}
"""

def classify_batch(messages: list, max_retries: int = 4) -> list:
    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
    prompt = BATCH_PROMPT.format(
        intents="\n".join(f"- {i}" for i in INTENTS),
        n=len(messages),
        messages=numbered
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()
            start, end = text.find("["), text.rfind("]")
            labels = json.loads(text[start:end+1])
            if len(labels) != len(messages):
                raise ValueError(f"Expected {len(messages)} labels, got {len(labels)}")
            return [l if l in INTENTS else "Other/Unclear" for l in labels]
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"  Batch error ({e}), retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)

    print(f"  Batch failed after {max_retries} attempts, marking all as Other/Unclear")
    return ["Other/Unclear"] * len(messages)
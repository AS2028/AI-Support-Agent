import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
from generate_reply import generate_grounded_reply

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

JUDGE_PROMPT = """You are evaluating an AI-generated customer support reply for quality.

Customer message: "{message}"
AI-generated reply: "{reply}"

Rate the reply on a scale of 1-5 for each criterion:
- relevance: does it actually address the customer's issue?
- tone: is it appropriately professional and empathetic?
- resolution: does it move toward actually solving the problem (not just acknowledging it)?

Respond with ONLY a JSON object like: {{"relevance": 4, "tone": 5, "resolution": 3}}
"""

def judge_reply(message: str, reply: str, max_retries: int = 4) -> dict:
    prompt = JUDGE_PROMPT.format(message=message, reply=reply)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()
            start, end = text.find("{"), text.rfind("}")
            return json.loads(text[start:end+1])
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"  Judge error ({e}), retrying in {wait_time}s...")
            time.sleep(wait_time)
    return {"relevance": None, "tone": None, "resolution": None}

if __name__ == "__main__":
    msg = "My package was supposed to arrive 3 days ago and there's still no update"
    reply = generate_grounded_reply(msg)
    print(f"MESSAGE: {msg}")
    print(f"REPLY: {reply}")
    scores = judge_reply(msg, reply)
    print(f"JUDGE SCORES: {scores}")
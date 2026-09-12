import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
pairs = pd.read_csv("data/amazon_pairs_final.csv")
embeddings = np.load("data/pair_embeddings.npy")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_similar(message: str, k: int = 3):
    """Find the k most similar past customer messages and their Amazon replies."""
    query_embedding = embed_model.encode([message])[0]
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    top_k_idx = np.argsort(similarities)[-k:][::-1]  # highest similarity first
    return pairs.iloc[top_k_idx]

REPLY_PROMPT = """You are an Amazon customer support agent. Here are {k} examples of how Amazon has resolved similar past issues:

{examples}

Now, using the SAME tone and resolution style shown above, draft a reply to this NEW customer message:
"{message}"

Respond with ONLY the reply text, nothing else.
"""

def generate_grounded_reply(message: str, k: int = 3) -> str:
    similar = retrieve_similar(message, k)
    examples = "\n\n".join(
        f"Customer: {row['customer_message']}\nAmazon: {row['amazon_reply']}"
        for _, row in similar.iterrows()
    )
    prompt = REPLY_PROMPT.format(k=k, examples=examples, message=message)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    test_messages = [
        "My package was supposed to arrive 3 days ago and there's still no update",
        "The headphones I ordered arrived with a cracked case",
    ]
    for msg in test_messages:
        print(f"CUSTOMER: {msg}")
        similar = retrieve_similar(msg, k=2)
        print("--- Retrieved similar past cases ---")
        for _, row in similar.iterrows():
            print(f"  [{row['predicted_intent']}] {row['customer_message'][:80]}")
        reply = generate_grounded_reply(msg)
        print(f"\nGENERATED REPLY: {reply}")
        print("=" * 60)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

pairs = pd.read_csv("data/amazon_pairs_classified.csv")

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(pairs["customer_message"])
y = pairs["predicted_intent"]

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X, y)

joblib.dump(model, "data/baseline_model.pkl")
joblib.dump(vectorizer, "data/baseline_vectorizer.pkl")
print("Trained and saved simple baseline model")

def simple_baseline(messages: list) -> list:
    X_new = vectorizer.transform(messages)
    return model.predict(X_new).tolist()

if __name__ == "__main__":
    test = ["Where is my order?", "I can't log into my account", "Any deals today?"]
    print(simple_baseline(test))
import pandas as pd

def trivial_baseline(messages: list) -> list:
    """Always predicts the single most common intent from our training data."""
    return ["Delivery Issue"] * len(messages)  # most common intent from our 300-pair corpus

if __name__ == "__main__":
    test = ["Where is my order?", "I can't log in", "Any deals today?"]
    print(trivial_baseline(test))
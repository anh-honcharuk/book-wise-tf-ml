import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification


MODEL_PATH = "models/genre_classifier"

GENRE_NAMES = [
    "Fantasy",
    "Romance",
    "Mystery",
    "Young Adult",
    "Fiction",
    "Historical",
    "Science Fiction",
    "Adventure",
    "Classics",
    "Thriller",
    "Horror",
    "Nonfiction",
    "Biography",
    "History",
    "Children",
    "Poetry",
    "Drama",
    "Comedy",
]


tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)


def predict_genre(description, threshold=0.3):
    tokens = tokenizer(
        description,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf",
    )

    outputs = model(tokens)
    probabilities = 1 / (1 + np.exp(-outputs.logits.numpy()[0]))

    results = sorted(
        zip(GENRE_NAMES, probabilities),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        (genre, float(probability))
        for genre, probability in results
        if probability >= threshold
    ]


if __name__ == "__main__":
    description = """
    A young wizard discovers a magical world and must fight
    an ancient evil threatening his friends and family.
    """

    results = predict_genre(description)

    print("Predicted genres:")

    for genre, probability in results:
        print(f"- {genre}: {probability:.2f}")
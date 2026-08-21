from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


dataset = load_dataset("pszemraj/goodreads-bookgenres")

books = dataset["train"].select(range(500))

book_ids = list(range(len(books)))
book_titles = books["Book"]
book_descriptions = books["Description"]

combined_text = [
    f"{title}. {description}"
    for title, description in zip(
        book_titles,
        book_descriptions,
    )
]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
)

tfidf_matrix = vectorizer.fit_transform(combined_text)

print("TF-IDF matrix shape:", tfidf_matrix.shape)


def recommend_similar(book_id, top_n=5):
    """
    Рекомендації для книги, яка вже є в dataset.
    """

    idx = book_ids.index(book_id)

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix,
    )[0]

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = [
        i
        for i in similar_indices
        if i != idx
    ]

    recommendations = []

    for i in similar_indices[:top_n]:
        recommendations.append(
            {
                "book_id": book_ids[i],
                "title": book_titles[i],
                "similarity": float(similarity_scores[i]),
            }
        )

    return recommendations


def recommend_by_description(description, top_n=5):
    """
    Рекомендації для нового опису книги,
    якого немає в dataset.
    """

    description_vector = vectorizer.transform(
        [description]
    )

    similarity_scores = cosine_similarity(
        description_vector,
        tfidf_matrix,
    )[0]

    similar_indices = similarity_scores.argsort()[::-1][:top_n]

    recommendations = []

    for i in similar_indices:
        recommendations.append(
            {
                "book_id": book_ids[i],
                "title": book_titles[i],
                "similarity": float(similarity_scores[i]),
            }
        )

    return recommendations


if __name__ == "__main__":

    selected_book_id = 0

    print("\nSelected book:")
    print(book_titles[selected_book_id])

    print("\nRecommended books:")

    recommendations = recommend_similar(
        selected_book_id,
        top_n=5,
    )

    for number, book in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"{number}. {book['title']} "
            f"(similarity: {book['similarity']:.3f})"
        )
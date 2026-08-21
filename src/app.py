import streamlit as st

from predict_genre import predict_genre
from recommender import recommend_by_description


class BookWiseApp:

    def run(self):
        st.title("📚 BookWise - Розумна система рекомендації книг")

        description = st.text_area(
            "Опис книги:",
            height=200,
            placeholder=(
                "A young wizard discovers a magical world "
                "and must fight an ancient evil..."
            ),
        )

        if st.button("Аналізувати та рекомендувати"):

            if not description.strip():
                st.warning("Будь ласка, введіть опис книги.")
                return

            # Класифікація жанрів
            st.subheader("Визначені жанри")

            genres = predict_genre(description)

            if genres:
                for genre, probability in genres:
                    st.write(
                        f"- **{genre}**: {probability:.2f}"
                    )
            else:
                st.write("Жанри не визначені.")

            # Рекомендації
            st.subheader("Рекомендовані книги")

            recommendations = recommend_by_description(
                description,
                top_n=5,
            )

            for number, book in enumerate(
                recommendations,
                start=1,
            ):
                st.write(
                    f"{number}. **{book['title']}** "
                    f"(similarity: {book['similarity']:.3f})"
                )


if __name__ == "__main__":
    app = BookWiseApp()
    app.run()
import os

import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizer, TFBertForSequenceClassification


class GenreClassifier:
    """
    BERT-модель для multi-label класифікації книг за жанрами.
    """

    def __init__(
        self,
        num_genres=18,
        max_length=128,
        model_name="bert-base-uncased",
    ):
        self.num_genres = num_genres
        self.max_length = max_length
        self.model_name = model_name

        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name
        )

        # BERT model
        self.model = TFBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_genres,
            from_pt=True,
        )

    def tokenize_data(self, example):
        """
        Токенізація опису книги.
        """

        tokens = self.tokenizer(
            example["Description"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "token_type_ids": tokens["token_type_ids"],
            "labels": example["Genres"],
        }

    def prepare_dataset(self, dataset, batch_size=8, shuffle=False):
        """
        Перетворення Hugging Face Dataset
        у TensorFlow Dataset.
        """

        tokenized_dataset = dataset.map(
            self.tokenize_data
        )

        tf_dataset = tokenized_dataset.to_tf_dataset(
            columns=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ],
            label_cols=["labels"],
            shuffle=shuffle,
            batch_size=batch_size,
        )

        return tf_dataset

    def compile(self):
        """
        Налаштування процесу навчання.
        """

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=2e-5
            ),
            loss=tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            ),
            metrics=[
                tf.keras.metrics.BinaryAccuracy()
            ],
        )

    def train(
        self,
        train_dataset,
        validation_dataset,
        epochs=1,
        batch_size=8,
    ):
        """
        Навчання моделі.
        """

        train_tf = self.prepare_dataset(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        validation_tf = self.prepare_dataset(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        self.compile()

        history = self.model.fit(
            train_tf,
            validation_data=validation_tf,
            epochs=epochs,
        )

        return history

    def evaluate(
        self,
        test_dataset,
        batch_size=8,
    ):
        """
        Оцінка моделі на тестовому dataset.
        """

        test_tf = self.prepare_dataset(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        results = self.model.evaluate(
            test_tf,
            return_dict=True,
        )

        return results

    def predict(self, description):
        """
        Передбачення жанрів для нового опису книги.
        """

        inputs = self.tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )

        outputs = self.model(inputs)

        # Для multi-label класифікації
        probabilities = tf.sigmoid(outputs.logits)

        return probabilities

    def save(self, path="models/genre_classifier"):
        """
        Збереження навченої моделі та tokenizer.
        """

        os.makedirs(
            path,
            exist_ok=True,
        )

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        print(f"Model saved to: {path}")

    @classmethod
    def load(
        cls,
        path="models/genre_classifier",
        num_genres=18,
        max_length=128,
    ):
        """
        Завантаження вже навченої моделі.
        """

        classifier = cls.__new__(cls)

        classifier.num_genres = num_genres
        classifier.max_length = max_length
        classifier.model_name = path

        classifier.tokenizer = BertTokenizer.from_pretrained(
            path
        )

        classifier.model = TFBertForSequenceClassification.from_pretrained(
            path,
            from_pt=True,
        )

        return classifier


# ============================================================
# Демонстрація роботи
# ============================================================

if __name__ == "__main__":

    print("Loading dataset...")

    dataset = load_dataset(
        "pszemraj/goodreads-bookgenres"
    )

    # Для першого запуску використовуємо невелику
    # частину train/validation, щоб навчання не тривало годинами.
    train_data = dataset["train"].select(
        range(500)
    )

    validation_data = dataset["validation"].select(
        range(100)
    )

    test_data = dataset["test"]

    print(
        f"Train: {len(train_data)}"
    )

    print(
        f"Validation: {len(validation_data)}"
    )

    print(
        f"Test: {len(test_data)}"
    )

    # ========================================================
    # Створюємо класифікатор
    # ========================================================

    classifier = GenreClassifier(
        num_genres=18,
        max_length=128,
    )

    # ========================================================
    # Навчання
    # ========================================================

    print("\nStarting training...")

    history = classifier.train(
        train_dataset=train_data,
        validation_dataset=validation_data,
        epochs=1,
        batch_size=8,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    print("\nEvaluating model...")

    results = classifier.evaluate(
        test_dataset=test_data,
        batch_size=8,
    )

    print("\nTest results:")

    for name, value in results.items():
        print(
            f"{name}: {value:.4f}"
        )

    # ========================================================
    # Збереження
    # ========================================================

    classifier.save(
        "models/genre_classifier"
    )

    print("\nDone!")
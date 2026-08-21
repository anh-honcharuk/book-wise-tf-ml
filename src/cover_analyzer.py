import tensorflow as tf
from tensorflow.keras import layers, models


class CoverAnalyzer:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.model = self.build_cnn_model()

    def build_cnn_model(self):
        """CNN для класифікації стилю обкладинки"""

        model = models.Sequential([
            layers.Input(
                shape=(
                    self.img_size[0],
                    self.img_size[1],
                    3,
                )
            ),

            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),

            layers.Dense(
                128,
                activation="relu",
            ),

            layers.Dropout(0.5),

            # 10 стилів, як у початковому завданні
            layers.Dense(
                10,
                activation="softmax",
            ),
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model


if __name__ == "__main__":
    analyzer = CoverAnalyzer()

    analyzer.model.summary()
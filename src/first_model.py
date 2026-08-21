import tensorflow as tf


# Дані
x = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
y = tf.constant([2, 4, 6, 8, 10], dtype=tf.float32)


# Модель
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])


# Налаштування навчання
model.compile(
    optimizer="sgd",
    loss="mse"
)


# Навчання
model.fit(
    x,
    y,
    epochs=100,
    verbose=0
)


# Prediction
prediction = model.predict(
    tf.constant([6.0]),
    verbose=0
)

print("Prediction:", prediction)
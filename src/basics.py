import tensorflow as tf


# 1. Tensor і tf.constant
x = tf.constant([1, 2, 3])

print(x)
print(x.shape)
print(x.dtype)


# 2. Операції над Tensor
y = tf.constant([4, 5, 6])

print(x + y)
print(x * y)


# 3. tf.Variable
weight = tf.Variable(2.0)

print("До:", weight)

weight.assign(5.0)

print("Після:", weight)


# 4. Операції над Variable
a = tf.Variable(2.0)
b = tf.Variable(3.0)

result = a * b

print(result)


# 5. GradientTape
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)

print("y =", y)
print("gradient =", gradient)
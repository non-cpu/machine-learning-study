import tensorflow as tf

from tensorflow import exp
from tensorflow.keras.layers import Dense

x = tf.random.normal(shape=(1, 5))

dense_sigmoid = Dense(units=1, activation='sigmoid')

y_tf = dense_sigmoid(x)

W, B = dense_sigmoid.get_weights()
z = tf.linalg.matmul(x, W) + B
y_man = 1 / (1 + exp(-z))

print(x.shape, x)
print(W.shape, W)
print(B.shape, B)

print(z)
print(y_tf)
print(y_man)

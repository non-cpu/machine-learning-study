import tensorflow as tf

from tensorflow.keras.layers import Dense

x = tf.constant([[10.]])

dense = Dense(units=1)

y_tf = dense(x)

W, B = dense.get_weights()

print(x.shape, x)
print(W.shape, W)
print(B.shape, B)

y_man = tf.linalg.matmul(x, W) + B

print(y_tf, y_man)

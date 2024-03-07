import tensorflow as tf

from tensorflow.keras.layers import Dense

x = tf.random.uniform(shape=(1, 10), minval=0, maxval=10)

dense = Dense(units=1)

y_tf = dense(x)

W, B = dense.get_weights()

print(x.shape, x)
print(W.shape, W)
print(B.shape, B)

print(y_tf)

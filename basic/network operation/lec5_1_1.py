# Shapes of Conv Layers

import tensorflow as tf

from tensorflow.keras.layers import Conv2D

N, n_H, n_W, n_C = 32, 28, 28, 5
n_filter = 10
k_size = 3

images = tf.random.uniform(shape=(N, n_H, n_W, n_C))

conv = Conv2D(filters=n_filter, kernel_size=k_size)

Y = conv(images)

W, B = conv.get_weights()

print(images.shape)
print(W.shape)
print(B.shape)
print(Y.shape)

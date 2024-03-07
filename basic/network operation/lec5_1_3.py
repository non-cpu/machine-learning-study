# Correlation with n-channel

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D

N, n_H, n_W, n_C = 1, 5, 5, 3
n_filter = 1
k_size = 3

images = tf.random.uniform(shape=(N, n_H, n_W, n_C))

conv = Conv2D(filters=n_filter, kernel_size=k_size)

y_tf = conv(images)

W, B = conv.get_weights()

images = tf.squeeze(images)
W = tf.squeeze(W)

y_man = np.zeros(shape=(n_H - k_size + 1, n_W - k_size + 1))

for i in range(n_H - k_size + 1):
    for j in range(n_W - k_size + 1):
        window = images[i:i+k_size, j:j+k_size, :]
        y_man[i, j] = np.sum(window * W) + B

print(tf.squeeze(y_tf))
print(y_man)

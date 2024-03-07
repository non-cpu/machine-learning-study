# 2D Max Pooling

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPool2D

N, n_H, n_W, n_C = 1, 5, 5, 1
f, s = 2, 1

x = tf.random.normal(shape=(N, n_H, n_W, n_C))
pool_max = MaxPool2D(pool_size=f, strides=s)
pooled_max = pool_max(x)

print(tf.squeeze(x))
print(tf.squeeze(pooled_max))

x = x.numpy().squeeze()
pooled_max_man = np.zeros(shape=(n_H - f + 1, n_W - f + 1))
for i in range(n_H - f + 1):
    for j in range(n_W - f + 1):
        window = x[i:i+f, j:j+f]
        pooled_max_man[i, j] = np.max(window)

print(pooled_max_man)

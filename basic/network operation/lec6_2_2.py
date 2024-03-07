# 2D Average Pooling

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import AveragePooling2D

N, n_H, n_W, n_C = 1, 5, 5, 1
f, s = 2, 1

x = tf.random.normal(shape=(N, n_H, n_W, n_C))
pool_avg = AveragePooling2D(pool_size=f, strides=s)
pooled_avg = pool_avg(x)

print(tf.squeeze(x))
print(tf.squeeze(pooled_avg))

x = x.numpy().squeeze()
pooled_avg_man = np.zeros(shape=(n_H - f + 1, n_W - f + 1))
for i in range(n_H - f + 1):
    for j in range(n_W - f + 1):
        window = x[i:i+f, j:j+f]
        pooled_avg_man[i, j] = np.mean(window)

print(pooled_avg_man)

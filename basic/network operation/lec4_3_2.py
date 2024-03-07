# BCE

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

N, n_feature = 100, 5
batch_size = 32

t_weights = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
t_bias = tf.constant([10], dtype=tf.float32)

X = tf.random.normal(shape=(N, n_feature))
Y = tf.reduce_sum(t_weights * X, axis=1) + t_bias
Y = tf.cast(Y > 5, dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(batch_size)

model = Dense(units=1, activation='sigmoid')
loss_object = BinaryCrossentropy()

for x, y in dataset:
    predictions = model(x)
    loss = loss_object(y, predictions)
    print(loss)

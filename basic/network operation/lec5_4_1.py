# Correlation with n-channel

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

n_neurons = [10, 20, 30]

model = Sequential()
model.add(Conv2D(filters=n_neurons[0], kernel_size=4, activation='relu'))
model.add(Conv2D(filters=n_neurons[1], kernel_size=4, activation='relu'))
model.add(Conv2D(filters=n_neurons[2], kernel_size=4, activation='relu'))

X = tf.random.uniform(shape=(32, 28, 28, 3))

predictions = model(X)

for layer in model.layers:
    W, B = layer.get_weights()
    print(W.shape, B.shape)

trainable_variables = model.trainable_variables
for trainable_var in trainable_variables:
    print(trainable_var.shape)

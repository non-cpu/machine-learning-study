import tensorflow as tf

from tensorflow.keras.layers import Dense

N, n_feature = 4, 10
X = tf.random.normal(shape=(N, n_feature))

n_neuron = 3
dense = Dense(units=n_neuron, activation='sigmoid')

Y = dense(X)

W, B = dense.get_weights()

print(X.shape)
print(W.shape)
print(B.shape)
print(Y.shape)

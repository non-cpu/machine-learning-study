import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

X = tf.random.normal(shape=(4, 10))

model = Sequential()
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=20, activation='sigmoid'))

Y = model(X)
print(Y.shape)

class TestModel(Model):
    def __init__(self, n_neurons):
        super(TestModel, self).__init__()
        self.n_neurons = n_neurons

        self.dense_layers = list()
        for n_neuron in self.n_neurons:
            self.dense_layers.append(Dense(units=n_neuron, activation='sigmoid'))

    def call(self, x):
        for dense in self.dense_layers:
            x = dense(x)
        return x

n_neurons = [3, 4, 5]
model = TestModel(n_neurons)

Y = model(X)
print(Y.shape)

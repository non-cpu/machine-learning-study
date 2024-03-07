import tensorflow as tf

from tensorflow import exp, maximum
from tensorflow.keras.layers import Activation

x = tf.random.normal(shape=(1, 5))

sigmoid = Activation('sigmoid')
tanh = Activation('tanh')
relu = Activation('relu')

y_sigmoid_tf = sigmoid(x)
y_tanh_tf = tanh(x)
y_relu_tf = relu(x)

y_sigmoid_man = 1 / (1 + exp(-x))
y_tanh_man = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
y_relu_man = maximum(0, x)

print(x)

print(y_sigmoid_tf)
print(y_sigmoid_man)

print(y_tanh_tf)
print(y_tanh_man)

print(y_relu_tf)
print(y_relu_man)

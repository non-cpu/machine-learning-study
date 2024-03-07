import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant

x = tf.constant([[10.]])

w, b = tf.constant(10.), tf.constant(20.)
w_init, b_init = Constant(w), Constant(b)

dense = Dense(units=1, kernel_initializer=w_init, bias_initializer=b_init)

y_tf = dense(x)

W, B = dense.get_weights()

print(x.shape, x)
print(W.shape, W)
print(B.shape, B)

print(y_tf)

# MLP + optimizer, scheduler

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plt.figure(figsize=(10, 10))
# for c in range(16):
#     plt.subplot(4, 4, c + 1)
#     plt.imshow(x_train[c].squeeze(), cmap='gray')

# plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0

train_size = int(len(x_train) * 0.9)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# print(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset))

train_batch_size = 1024
val_batch_size = 128
test_batch_size = 1024

train_dataloader = train_dataset.batch(train_batch_size, drop_remainder=True)
val_dataloader = val_dataset.batch(val_batch_size, drop_remainder=True)
test_dataloader = test_dataset.batch(test_batch_size, drop_remainder=True)

class MLP(tf.keras.Model):
    def __init__(self, h1_dim, h2_dim, out_dim, dropout_prob):
        super().__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(units=h1_dim, activation='relu')
        self.dense2 = Dense(units=h2_dim, activation='relu')
        self.dense3 = Dense(units=out_dim)
        self.dropout = Dropout(rate=dropout_prob)

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        out = self.dense3(x)
        return out
    
    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            outputs = self(images, training=True)
            loss = self.compiled_loss(labels, outputs)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        preds = tf.argmax(outputs, axis=1)
        self.compiled_metrics.update_state(labels, preds)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        images, labels = data

        outputs = self(images)
        self.compiled_loss(labels, outputs)
        preds = tf.argmax(outputs, axis=1)
        self.compiled_metrics.update_state(labels, preds)

        return {m.name: m.result() for m in self.metrics}

class LinearWarmLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_peak, warmup_end_steps):
        self.lr_peak = lr_peak
        self.warmup_end_steps = warmup_end_steps

    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        lr_peak = tf.cast(self.lr_peak, tf.float32)
        warmup_end_steps = tf.cast(self.warmup_end_steps, tf.float32)

        return tf.cond(
            step_float < warmup_end_steps,
            lambda: tf.multiply(lr_peak, tf.divide(tf.math.maximum(step_float, tf.cast(tf.constant(1), tf.float32)), warmup_end_steps)),
            lambda: lr_peak
        )

n_class = 10
max_epoch = 50
learning_rate = 1e-3

loss_function = SparseCategoricalCrossentropy(from_logits=True)

schedule = LinearWarmLRSchedule(lr_peak=learning_rate, warmup_end_steps=200)

# optimizer = Adam(learning_rate=learning_rate) # 19, 0.3506, 0.9894
# optimizer = Adam(learning_rate=schedule) # 31, 0.3462, 0.9899

# optimizer = SGD(learning_rate=learning_rate) # 50, 1.0920, 0.9533
# optimizer = SGD(learning_rate=schedule) # 50, 1.0685, 0.9514

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate) # 35, 0.3370, 0.9897

# optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=learning_rate) # 28, 0.3368, 0.9900
# optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=schedule) # 34, 0.3283, 0.9905

model = MLP(128, 64, n_class, dropout_prob=0.3)

model.build((1, 28, 28))
model.summary()

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=['accuracy']
)

model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=max_epoch,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)

model.evaluate(test_dataloader)

test_labels = np.zeros(shape=(len(test_dataloader) * test_batch_size))
test_outputs = np.zeros(shape=(len(test_dataloader) * test_batch_size, n_class))

for i, (images, labels) in enumerate(test_dataloader):
    test_labels[i*test_batch_size:(i+1)*test_batch_size] = labels
    test_outputs[i*test_batch_size:(i+1)*test_batch_size] = tf.nn.softmax(model(images))

auc_score = roc_auc_score(test_labels, test_outputs, average='macro', multi_class='ovo')
print(f"auc_score: {auc_score}")

for i in range(n_class):
    fpr, tpr, _ = roc_curve(test_labels, test_outputs[:, i], pos_label=i)
    plt.plot(fpr, tpr, linestyle='dashed', label=str(i))

plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='best')
plt.show()

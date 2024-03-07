import numpy as np
import matplotlib.pyplot as plt

from termcolor import colored

from tensorflow.keras.datasets.mnist import load_data

plt.style.use('seaborn')

(train_images, train_labels), test_ds = load_data()

n_data = train_images.shape[0]
n_feature = train_images.shape[1] * train_images.shape[2]

lr = 0.03

epochs, b_size = 20, 128

units = [64, 32, 10]

n_batch = n_data // b_size

W1 = np.random.normal(0, 1, (n_feature, units[0]))
B1 = np.zeros(units[0])

W2 = np.random.normal(0, 1, (units[0], units[1]))
B2 = np.zeros(units[1])

W3 = np.random.normal(0, 1, (units[1], units[2]))
B3 = np.zeros(units[2])

# print(colored("W/B shapes", 'green'))
# print(f'W1/B1: {W1.shape}/{B1.shape}')
# print(f'W2/B2: {W2.shape}/{B2.shape}')
# print(f'W3/B3: {W3.shape}/{B3.shape}')

losses, accs = list(), list()

for epoch in range(epochs):
    n_correct, n_data = 0, 0

    for b_idx in range(n_batch):
        start_idx = b_idx * b_size
        end_idx = (b_idx + 1) * b_size

        images = train_images[start_idx:end_idx, ...]

        X = images.reshape(b_size, -1)
        Y = train_labels[start_idx:end_idx, ...]

        Z1 = X @ W1 + B1
        A1 = 1 / (1 + np.exp(-Z1))

        Z2 = A1 @ W2 + B2
        A2 = 1 / (1 + np.exp(-Z2))

        L = A2 @ W3 + B3

        Pred = np.exp(L) / np.sum(np.exp(L), axis=1, keepdims=True)

        J = np.mean(-np.log(Pred[np.arange(b_size), Y]))

        losses.append(J)

        Pred_label = np.argmax(Pred, axis=1)

        n_correct += np.sum(Pred_label == Y)
        n_data += b_size

        labels = Y.copy()
        Y = np.zeros_like(Pred)
        Y[np.arange(b_size), labels] = 1

        dL = -1 / b_size * (Y - Pred)
        dA2 = dL @ W3.T
        dW3 = A2.T @ dL
        dB3 = np.sum(dL, axis=0)

        dZ2 = dA2 * A2 * (1 - A2)
        dA1 = dZ2 @ W2.T
        dW2 = A1.T @ dZ2
        dB2 = np.sum(dZ2, axis=0)

        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = X.T @ dZ1
        dB1 = np.sum(dZ1, axis=0)

        W3, B3 = W3 - lr * dW3, B3 - lr * dB3
        W2, B2 = W2 - lr * dW2, B2 - lr * dB2
        W1, B1 = W1 - lr * dW1, B1 - lr * dB1

    accs.append(n_correct / n_data)

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(losses)
axes[1].plot(accs)

plt.show()

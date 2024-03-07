import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
plt.style.use('seaborn')

N, n_feature = 100, 3

epochs, b_size = 100, 32

lr = 0.01

t_W = np.random.uniform(-1, 1, (n_feature, 1))
t_b = np.random.uniform(-1, 1, (1, 1))

W = np.random.uniform(-1, 1, (n_feature, 1))
b = np.random.uniform(-1, 1, (1, 1))

n_batch = N // b_size

x_data = np.random.normal(0, 1, (N, n_feature))
y_data = x_data @ t_W + t_b
y_data = (y_data > 0).astype(int)

J_track, acc_track = list(), list()

for epoch in range(epochs):
    for b_idx in range(n_batch):
        X = x_data[b_idx * b_size:(b_idx+1) * b_size, ...]
        Y = y_data[b_idx * b_size:(b_idx+1) * b_size, ...]

        Z = X @ W + b
        Pred = 1 / (1 + np.exp(-Z))
        J0 = -(Y * np.log(Pred) + (1 - Y) * np.log(1 - Pred))
        J = np.mean(J0)
        J_track.append(J)

        Pred_ = (Pred > 0.5).astype(int)        
        n_correct = np.sum((Pred_ == Y).astype(int))
        acc = n_correct / b_size
        acc_track.append(acc)

        dJ_dJ0 = 1 / b_size * np.ones((1, b_size))
        dJ0_dPred = np.diag(((Pred - Y) / (Pred * (1 - Pred))).flatten())
        dPred_dZ = np.diag((Pred * (1 - Pred)).flatten())
        dPred_dW = X
        dPred_dB = np.ones((b_size, 1))

        dJ_dPred = dJ_dJ0 @ dJ0_dPred
        dJ_dZ = dJ_dPred @ dPred_dZ
        dJ_dW = dJ_dPred @ dPred_dW
        dJ_db = dJ_dPred @ dPred_dB

        W = W - lr * dJ_dW.T
        b = b - lr * dJ_db

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(J_track)
axes[1].plot(acc_track)

plt.show()

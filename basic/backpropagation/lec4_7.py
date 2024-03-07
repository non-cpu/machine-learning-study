import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
plt.style.use('seaborn')

N, n_feature = 1000, 5
lr = 0.03

t_W = np.random.uniform(-1, 1, (n_feature, 1))
t_b = np.random.uniform(-1, 1, (1))

W = np.random.uniform(-1, 1, (n_feature, 1))
b = np.random.uniform(-1, 1, (1, 1))

x_data = np.random.randn(N, n_feature)

y_data = x_data @ t_W + t_b
y_data = 1 / (1 + np.exp(-y_data))
y_data = (y_data > 0.5).astype(np.int)

J_track, W_track, b_track, acc_track = list(), list(), list(), list()

n_correct = 0

for data_idx, (X, y) in enumerate(zip(x_data, y_data)):
    W_track.append(W)
    b_track.append(b)

    X = X.reshape(1, -1)

    z = X @ W + b
    pred = 1 / (1 + np.exp(-z))
    J = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    J_track.append(J.squeeze())

    pred_ = (pred > 0.5).astype(np.int).squeeze()

    if pred_ == y:
        n_correct += 1

    acc_track.append(n_correct / (data_idx + 1))

    dJ_dpred = (pred - y) / (pred * (1 - pred))
    dpred_dz = pred * (1 - pred)
    dz_dW = X.reshape(1, -1)
    dz_db = 1

    dJ_dz = dJ_dpred * dpred_dz
    dJ_dW = dJ_dz * dz_dW
    dJ_db = dJ_dz * dz_db

    W = W - lr * dJ_dW.T
    b = b - lr * dJ_db

fig, axes = plt.subplots(2, 1, figsize=(10, 5))

axes[0].plot(J_track)
axes[0].set_ylabel('BCEE', fontsize=15)
axes[0].tick_params(labelsize=10)

axes[1].plot(acc_track)
axes[1].set_ylabel('ACC', fontsize=15)
axes[1].tick_params(labelsize=10)

plt.show()

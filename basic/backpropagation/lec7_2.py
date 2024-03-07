import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

np.random.seed(0)
plt.style.use('seaborn')

N, n_feature = 300, 5

epochs, b_size = 30, 32

lr = 0.03

t_W = np.random.uniform(-1, 1, (n_feature, 1))
t_b = np.random.uniform(-1, 1, (1, 1))

W = np.random.uniform(-1, 1, (n_feature, 1))
b = np.random.uniform(-1, 1, (1, 1))

n_batch = N // b_size

x_data = np.random.randn(N, n_feature)
y_data = x_data @ t_W + t_b

J_track, W_track, b_track = list(), list(), list()

for epoch in range(epochs):
    for b_idx in range(n_batch):
        W_track.append(W)
        b_track.append(b)

        X = x_data[b_idx * b_size:(b_idx+1) * b_size, ...]
        Y = y_data[b_idx * b_size:(b_idx+1) * b_size, ...]

        Pred = X @ W + b
        J0 = (Y - Pred) ** 2
        J = np.mean(J0)
        J_track.append(J)

        dJ_dJ0 = 1 / b_size * np.ones((1, b_size))
        dJ0_dPred = np.diag(-2 * (Y - Pred).flatten())
        dPred_dW = X
        dPred_dB = np.ones((b_size, 1))

        dJ_dPred = dJ_dJ0 @ dJ0_dPred
        dJ_dW = dJ_dPred @ dPred_dW
        dJ_db = dJ_dPred @ dPred_dB

        W = W - lr * dJ_dW.T
        b = b - lr * dJ_db

W_track = np.hstack(W_track)
b_track = np.concatenate(b_track)

cmap = cm.get_cmap('tab10', n_feature)
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(J_track)

for w_idx, (t_w, w_track) in enumerate(zip(t_W, W_track)):
    axes[1].axhline(y=t_w, color=cmap(w_idx), linestyle=':')
    axes[1].plot(w_track, color=cmap(w_idx))

axes[1].axhline(y=t_b, color='black', linestyle=':')
axes[1].plot(b_track, color='black')

plt.show()

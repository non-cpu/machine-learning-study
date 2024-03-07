from turtle import color
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

np.random.seed(1)
plt.style.use('seaborn')

# set params
N = 400
lr = 0.01
t_w, t_b = 5, -3
w, b = np.random.uniform(-3, 3, 2)

x_data = np.random.randn(N)
y_data = x_data * t_w + t_b

# y_data += 0.5 * np.random.randn(N) # noise

cmap = cm.get_cmap('rainbow', lut=N)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(x_data, y_data)

x_range = np.array([x_data.min(), x_data.max()])

J_track, w_track, b_track = list(), list(), list()

for data_idx, (x, y) in enumerate(zip(x_data, y_data)):
    w_track.append(w)
    b_track.append(b)

    y_range = w * x_range + b
    ax.plot(x_range, y_range, color=cmap(data_idx), alpha=0.5)

    pred = x * w + b
    J = (y - pred) ** 2

    J_track.append(J)

    dJ_dpred = -2 * (y - pred)

    dpred_dw = x
    dpred_db = 1

    dJ_dw = dJ_dpred * dpred_dw
    dJ_db = dJ_dpred * dpred_db

    w = w - lr * dJ_dw
    b = b - lr * dJ_db

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(J_track)
axes[0].set_ylabel('MSE', fontsize=15)
axes[0].tick_params(labelsize=10)

axes[1].plot(w_track, color='darkred')
axes[1].plot(b_track, color='darkblue')

axes[1].axhline(y=t_w, color='darkred', linestyle=':')
axes[1].axhline(y=t_b, color='darkBlue', linestyle=':')
axes[1].tick_params(labelsize=10)

plt.show()

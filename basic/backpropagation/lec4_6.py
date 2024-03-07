import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

np.random.seed(1)
plt.style.use('seaborn')

N = 3000
lr = 0.03

t_w = np.random.uniform(-3, 3, (1))
t_b = np.random.uniform(-3, 3, (1))

w = np.random.uniform(-3, 3, (1))
b = np.random.uniform(-3, 3, (1))

db = -(t_b / t_w)
x_data = np.random.normal(db, 1, (N))
y_data = x_data * t_w + t_b
y_data = (x_data > db).astype(np.int)

# x_data += 0.2 * np.random.normal(0, 1, (N)) # noise

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(x_data, y_data)

cmap = cm.get_cmap('rainbow', lut=N)

J_track, w_track, b_track = list(), list(), list()

x_range = np.linspace(x_data.min(), x_data.max(), 100)

for data_idx, (x, y) in enumerate(zip(x_data, y_data)):
    w_track.append(w)
    b_track.append(b)

    y_range = w * x_range + b
    y_range = 1 / (1 + np.exp(-y_range))

    ax.plot(x_range, y_range, color=cmap(data_idx), alpha=0.3)

    z = x * w + b

    pred = 1 / (1 + np.exp(-z))
    J = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    J_track.append(J)

    dJ_dpred = (pred - y) / (pred * (1 - pred))
    dpred_dz = pred * (1 - pred)
    dz_dw = x
    dz_db = 1

    dJ_dz = dJ_dpred * dpred_dz
    dJ_dw = dJ_dz * dz_dw
    dJ_db = dJ_dz * dz_db

    w = w - lr * dJ_dw
    b = b - lr * dJ_db

fig, axes = plt.subplots(2, 1, figsize=(10, 5))

axes[0].plot(J_track)
axes[0].set_ylabel('BCEE', fontsize=15)
axes[0].tick_params(labelsize=10)

axes[1].axhline(y=t_w, color='darkred', linestyle=':')
axes[1].plot(w_track, color='darkred')
axes[1].axhline(y=t_b, color='darkblue', linestyle=':')
axes[1].plot(b_track, color='darkblue')
axes[1].tick_params(labelsize=10)

w_track = np.array(w_track)
b_track = np.array(b_track)

db_track = -b_track / w_track

fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(db, color='black', linestyle=':')
ax.plot(db_track, color='black')

plt.show()

# The Graphs of Odds and Logit

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

p_np = np.linspace(0.01, 0.99, 100)
p_tf = tf.linspace(0.01, 0.99, 100)

odds_np = p_np / (1 - p_np)
odds_tf = p_tf / (1 - p_tf)

logit_np = np.log(odds_np)
logit_tf = tf.math.log(odds_tf)

plt.style.use('seaborn')

fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(p_tf, odds_tf)
axes[1].plot(p_tf, logit_tf)

xticks = np.arange(0, 1.2, 0.2)

axes[0].tick_params(labelsize=10)
axes[0].set_xticks(xticks)
axes[0].set_ylabel('Odds', fontsize=15, color='darkblue')
axes[1].tick_params(labelsize=10)
axes[1].set_xticks(xticks)
axes[1].set_ylabel('Logit', fontsize=15, color='darkblue')
axes[1].set_xlabel('Probability', fontsize=15, color='darkblue')

plt.show()

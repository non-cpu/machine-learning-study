import tensorflow as tf
import numpy as np
import collections
import random
import time

from game import Game

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MAX_EPISODE = 10000
TARGET_UPDATE_INTERVAL = 1000
TRAIN_INTERVAL = 4
OBSERVE = 100

NUM_ACTION = 3
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10

class DQN:
	REPLAY_MEMORY = 10000
	BATCH_SIZE = 32
	GAMMA = 0.99
	STATE_LEN = 4

	def __init__(self, width, height, n_action):
		self.n_action = n_action
		self.width = width
		self.height = height
		self.memory = collections.deque()
		self.state = None
		
		self.Q = self._build_network()
		self.target_Q = self._build_network()

	def _build_network(self):
		inputs = tf.keras.layers.Input(shape=(self.width, self.height, self.STATE_LEN,))

		layer1 = tf.keras.layers.Conv2D(32, [4, 4], padding='same', activation=tf.nn.relu)(inputs)
		layer2 = tf.keras.layers.Conv2D(64, [2, 2], padding='same', activation=tf.nn.relu)(layer1)

		layer3 = tf.keras.layers.Flatten()(layer2)

		layer4 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(layer3)
		action = tf.keras.layers.Dense(self.n_action, activation=None)(layer4)

		return tf.keras.Model(inputs=inputs, outputs=action)

	def update_target_network(self):
		self.target_Q.set_weights(self.Q.get_weights())

	def get_action(self):
		state = np.expand_dims(self.state, 0)

		Q_value = self.Q(state, training=False)

		action = np.argmax(Q_value[0])

		return action

	def init_state(self, state):
		state = [state for _ in range(self.STATE_LEN)]
		self.state = np.stack(state, axis=2)

	def remember(self, state, action, reward, terminal):
		next_state = np.reshape(state, (self.width, self.height, 1))
		next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

		self.memory.append((self.state, next_state, action, reward, terminal))

		if len(self.memory) > self.REPLAY_MEMORY:
			self.memory.popleft()

		self.state = next_state

	def _sample_memory(self):
		sample_memory = random.sample(self.memory, self.BATCH_SIZE)
		
		state = [memory[0] for memory in sample_memory]
		next_state = [memory[1] for memory in sample_memory]
		action = [memory[2] for memory in sample_memory]
		reward = [memory[3] for memory in sample_memory]
		terminal = [memory[4] for memory in sample_memory]

		return state, next_state, action, reward, terminal

	def train(self):
		state, next_state, action, reward, terminal = self._sample_memory()

		target_Q_value = self.target_Q(np.array(next_state), training=False)
		target_Q_value = np.where(terminal, reward, self.GAMMA * np.amax(target_Q_value, axis=1) + reward)

		one_hot = tf.one_hot(action, self.n_action, on_value=1.0, off_value=0.0)
		
		with tf.GradientTape() as tape:
			Q_value = self.Q(np.array(state))
			Q_action = tf.reduce_sum(tf.multiply(Q_value, one_hot), axis=1)

			loss = tf.reduce_mean(tf.square(target_Q_value - Q_action))

		grads = tape.gradient(loss, self.Q.trainable_variables)
		tf.keras.optimizers.Adam(1e-4).apply_gradients(zip(grads, self.Q.trainable_variables))

def train():
	game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
	brain = DQN(SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

	brain.update_target_network()

	epsilon = 1.0
	time_step = 0
	total_reward_list = []

	for episode in range(MAX_EPISODE):
		terminal = False
		total_reward = 0

		state = game.reset()
		brain.init_state(state)

		while not terminal:
			if np.random.rand() < epsilon:
				action = random.randrange(NUM_ACTION)
			else:
				action = brain.get_action()

			if episode > OBSERVE:
				epsilon -= 1 / 1000.

			state, reward, terminal = game.step(action)
			total_reward += reward

			brain.remember(state, action, reward, terminal)

			if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
				brain.train()

			if time_step % TARGET_UPDATE_INTERVAL == 0:
				brain.update_target_network()

			time_step += 1

		print('episode: %d, score: %d' % (episode + 1, total_reward))

		total_reward_list.append(total_reward)
		
		if episode % 10 == 0:
			#summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
			#writer.add_summary(summary, time_step)
			total_reward_list = []
		
		if episode % 100 == 0:
			pass
            # saver.save(sess, 'model/dqn.ckpt', global_step=time_step)

if __name__ == '__main__':
	# tf.compat.v1.app.run()

	train()

import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN

tf.compat.v1.disable_eager_execution()

tf.compat.v1.flags.DEFINE_bool('train', False, 'train_mode')
FLAGS = tf.compat.v1.flags.FLAGS

MAX_EPISODE = 10000
TARGET_UPDATE_INTERVAL = 1000
TRAIN_INTERVAL = 4
OBSERVE = 100

NUM_ACTION = 3
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10

def train():
	print('wake up the brain...')
	sess = tf.compat.v1.Session()

	game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
	brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

	rewards = tf.compat.v1.placeholder(tf.float32, [None])
	tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

	saver = tf.compat.v1.train.Saver()
	sess.run(tf.compat.v1.global_variables_initializer())

	#writer = tf.compat.v1.summary.FileWriter('logs', sess.graph)
	#summary_merged = tf.compat.v1.summary.merge_all()

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
			saver.save(sess, 'model/dqn.ckpt', global_step=time_step)

def replay():
	print('wake up the brain...')
	sess = tf.compat.v1.Session()

	game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
	brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

	saver = tf.compat.v1.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint('model'))

	for episode in range(MAX_EPISODE):
		terminal = False
		total_reward = 0

		state = game.reset()
		brain.init_state(state)

		while not terminal:
			action = brain.get_action()

			state, reward, terminal = game.step(action)
			total_reward += reward

			brain.remember(state, action, reward, terminal)

			# time.sleep(0.1)

		print('episode: %d, score: %d' % (episode + 1, total_reward))

def main(_):
	if FLAGS.train:
		train()
	else:
		replay()

if __name__ == '__main__':
	# tf.compat.v1.app.run()

	train()
	# replay()

import tensorflow as tf
import numpy as np
import collections
import random
import time

from collections import deque

from game import Game

gpu = tf.config.experimental.list_physical_devices('GPU') # 내 컴에 장착된 GPU를 list로 반환
try:
    tf.config.experimental.set_memory_growth(gpu[0], True) # GPU Memory Growth를 Enable
except RuntimeError as e:
    print(e)

NUM_ACTION = 3
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10

max_episode = 100000

target_update_interval = 1000
train_interval = 4  # Train the model after 4 actions

update_after_actions = train_interval   # tmp

steps_per_episode = 10000

memory_length = 10000
batch_size = 32
gamma = 0.99  # Discount factor for past rewards

memory = deque()
state_s = None

num_actions = NUM_ACTION

state_length = 4	# 한번에 볼 프레임의 수

episode_reward_history = []

epsilon_random_frames = 5000 # 50000   # Number of frames to take random action and observe output
epsilon_greedy_frames = 10000 # 1000000.0   # Number of frames for exploration

# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)

# Using huber loss for stability
loss_function = tf.keras.losses.Huber()
# loss_function = tf.keras.losses.MeanSquaredError()

def remember(state, action, reward, terminal):
    global state_s
    next_state = np.reshape(state, (SCREEN_WIDTH, SCREEN_HEIGHT, 1))
    next_state = np.append(state_s[:, :, 1:], next_state, axis=2)

    memory.append((state_s, next_state, action, reward, terminal))

    if len(memory) > memory_length:
        memory.popleft()

    state = next_state

def _sample_replay_buffer():
    sample_memory = random.sample(memory, batch_size)
		
    state = [_memory[0] for _memory in sample_memory]
    next_state = [_memory[1] for _memory in sample_memory]
    action = [_memory[2] for _memory in sample_memory]
    reward = [_memory[3] for _memory in sample_memory]
    terminal = [_memory[4] for _memory in sample_memory]

    return state, next_state, action, reward, terminal

def init_state(state):
    global state_s
    state_s = [state for _ in range(state_length)]
    state_s = np.stack(state_s, axis=2)

def create_q_model():
    inputs = tf.keras.layers.Input(shape=(SCREEN_WIDTH, SCREEN_HEIGHT, state_length))

    layer1 = tf.keras.layers.Conv2D(32, [4, 4], padding='same', activation=tf.nn.relu)(inputs)
    layer2 = tf.keras.layers.Conv2D(64, [2, 2], padding='same', activation=tf.nn.relu)(layer1)

    layer3 = tf.keras.layers.Flatten()(layer2)

    layer4 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(layer3)
    action = tf.keras.layers.Dense(num_actions, activation=None)(layer4)

    return tf.keras.Model(inputs=inputs, outputs=action)

def train():
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.0  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken

    frame_count = 0
    episode_count = 0
    running_reward = 0

    env = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)

    model = create_q_model()
    model_target = create_q_model()

    #state, next_state, action, reward, terminal = _sample_replay_buffer()

    #brain.update_target_network()

    #epsilon = 1.0

    for episode in range(max_episode):
        state = env.reset()    # numpy array
        init_state(state)

        episode_reward = 0

        for _ in range(1, steps_per_episode):
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state_s)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment            
            state_next, reward, done = env.step(action)
            #state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            remember(state_next, action, reward, done)

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(memory) > batch_size:

                # Get indices of samples for replay buffers
                # indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample, state_next_sample, action_sample, rewards_sample, done_sample = _sample_replay_buffer()

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(np.array(state_next_sample))
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                done_sample = np.array(done_sample)
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions, on_value=1.0, off_value=0.0)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(np.array(state_sample))

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % target_update_interval == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(memory) > memory_length:
                memory.popleft()

            if done:
                break
        
        print('episode: {}, score: {}'.format(episode + 1, episode_reward))
        
        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward > 40:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break

if __name__ == '__main__':
    train()

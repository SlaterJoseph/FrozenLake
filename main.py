import numpy as np
import gymnasium as gym
import random

base_env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')  # This was we don't keep moving until we hit
# an end (either hole or wall)

action_space_size = base_env.action_space.n
state_space_size = base_env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

total_episodes = 11000  # Times played
learning_rate = 0.2  #
max_steps = 100  # Prevents infinite loops
gamma = 0.99  #
epsilon = 1  # Higher --> More exploration, Lower --> More exploitation (table use)
max_epsilon = 1  #
min_epsilon = 0.01  # To prevent 0 explorations
decay_rate = 0.001  # To lean more to exploitation as time progresses

rewards = list()

human = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')  # This was we don't keep moving until we hit
rgb = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')  # This was we don't keep moving until we hit


for episode in range(total_episodes):
    if episode == 1000 or episode == 3000 or episode == 5000 or episode == 7000 or episode == 9000 or episode == 11000:
        env = human
    else:
        env = rgb
    state, info = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):  # Sets the max step range up
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(q_table[state, :])  # Exploit
        else:
            action = env.action_space.sample()  # Explore

        new_state, reward, terminated, done, info = env.step(action)
        max_new_state = np.max(q_table[new_state, :])
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * max_new_state -
                                                                           q_table[state, action])
        total_rewards += reward
        state = new_state

        if episode % 1000 == 0:
            env.render()

        if done:
            print(f"Episode: {episode}/{total_episodes}, score: {sum(rewards) / total_episodes}, e: {epsilon}")
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

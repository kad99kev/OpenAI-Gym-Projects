import gym
import numpy as np
from IPython.display import clear_output
from time import sleep

env = gym.make('FrozenLake-v0')


action_space_size = env.action_space.n
state_space_size = env.observation_space.n
print(action_space_size)
print(state_space_size)

q_table = np.zeros(shape=(state_space_size, action_space_size))
print(q_table)
#
# Hyperparameters
TOTAL_EPISODES = 5_000 #Number of epsiodes to train the algorithm
MAX_STEPS = 100 #Max steps an agent can take during an episode

LEARNING_RATE = 0.7
GAMMA = 0.6 # Discount (close to 0 makes it greedy, close to 1 considers long term)

# Exploration Parameters
epsilon = 1
MAX_EPSILON = 1
MIN_EPSILON = 0.001
DECAY_RATE = 0.001

env.reset()
goal_list = []
total_rewards = 0


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print('\n*********************')
        print(f"Episode: {frame['episode']}")
        print(frame['frame'])
        print(f'Timestep: {i + 1}')
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(0.1)

for episode in range(TOTAL_EPISODES):

    state = env.reset()
    frames = []
    step = 0
    done = False

    for step in range(MAX_STEPS):

        if np.random.random() > epsilon: #This is exploitation
            action = np.argmax(q_table[state, :]) #Current state, max value
        else: #This is exploration
            action = np.random.randint(0, action_space_size)

        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        frames.append({
            'episode': episode,
            'frame': env.render(mode='ansi'),
            'state': new_state,
            'action': action,
            'reward': reward
        })

        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(q_table[new_state, :]))

        if reward == 1:
            print_frames(frames)

        if done:
            break

        state = new_state

    if MIN_EPSILON <= epsilon <= MAX_EPSILON:
        epsilon -= DECAY_RATE

env.close()

print(q_table)
print(f'Total Score after running {TOTAL_EPISODES} episodes: {total_rewards}')

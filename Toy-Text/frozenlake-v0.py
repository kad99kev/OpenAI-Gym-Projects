import gym
import numpy as np
from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')


action_space_size = env.action_space.n
state_space_size = env.observation_space.n
print(f"Number of actions available: {action_space_size}")
print(f"Number of states defined: {state_space_size}")
print(f"Therefore, Q-Table with {action_space_size} columns and {state_space_size} rows will be created")

q_table = np.random.random([state_space_size, action_space_size])
#OR
# q_table = np.zeros([state_space_size, action_space_size])
print(f"Q-Table shape: {q_table.shape}")
sleep(5)

# Hyperparameters
TOTAL_EPISODES = 5_000 #Number of epsiodes to train the algorithm
MAX_STEPS = 100 #Max steps an agent can take during an episode

LEARNING_RATE = 0.1
GAMMA = 0.95 # Discount (close to 0 makes it greedy, close to 1 considers long term)

# Exploration Parameters
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = TOTAL_EPISODES // 2
DECAY_RATE = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

env.reset()
history = {'steps': [], 'episode_number': []}
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
    history['steps'].append(i + 1)
    history['episode_number'].append(frame['episode'])

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

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= DECAY_RATE

env.close()

print(f"\nQ-Table after {TOTAL_EPISODES} episodes")
print(q_table)
print(f'Total Score after running {TOTAL_EPISODES} episodes: {total_rewards}')

# Number of steps
plt.plot(history['episode_number'], history['steps'], label = 'steps')
plt.legend(loc = 4)
plt.show()

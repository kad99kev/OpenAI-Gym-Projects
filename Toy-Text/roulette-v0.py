import gym
import numpy as np
from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt

env = gym.make('Roulette-v0')

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
MAX_STEPS = 150 #Max steps an agent can take during an episode

LEARNING_RATE = 0.1
GAMMA = 0.95 # Discount (close to 0 makes it greedy, close to 1 considers long term)

# Exploration Parameters
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = TOTAL_EPISODES // 2
DECAY_RATE = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def print_frames(frames):
    total_reward = 0
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print('\n*********************')
        print(f"Episode: {frame['episode']}")
        print(f'Round: {i + 1}')
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        total_reward += frame['reward']
        print(f"Total reward so far: {total_reward}")
        sleep(0.1)


def edit_reward(info):
    print(info)

env.reset()
history = {'steps': [], 'total_score': [], 'episode_number': []}
best_frames = []
high_score = 0

for episode in range(TOTAL_EPISODES):

    state = env.reset()
    step = 0
    frames = []
    current_score = 0
    done = False

    for step in range(MAX_STEPS):

        if np.random.random() > epsilon: #This is exploitation
            action = np.argmax(q_table[state, :]) #Current state, max value
        else: #This is exploration
            action = np.random.randint(0, action_space_size)

        new_state, reward, done, info = env.step(action)

        current_score += reward

        frames.append({
            'episode': episode,
            'action': action,
            'reward': reward
        })

        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(q_table[new_state, :]))

        if done:

            history['steps'].append(step + 1)
            history['total_score'].append(current_score)
            history['episode_number'].append(episode)

            if current_score >= high_score:
                high_score = current_score
                best_frames = frames.copy()
            break

        state = new_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= DECAY_RATE

env.close()
print_frames(best_frames)
print(f"\nQ-Table after {TOTAL_EPISODES} episodes")
print(q_table)
print(f'The best score after running {TOTAL_EPISODES} episodes: {high_score}')

# Total score
plt.subplot(2, 1, 1)
plt.scatter(history['episode_number'], history['total_score'], s = 5, label = 'score', color = 'blue')
plt.xlabel('Number of episodes')
plt.ylabel('Score')
plt.legend(loc = 4)

# Number of steps
plt.subplot(2, 1, 2)
plt.scatter(history['episode_number'], history['steps'], s = 5, label = 'steps', color = 'red')
plt.xlabel('Number of episodes')
plt.ylabel('Number of steps')
plt.legend(loc = 4)
plt.show()

import gym
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.2
GAMMA = 0.95 # Discount (close to 0 makes it greedy, close to 1 considers long term)

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

class FeatureTransformer:
    def __init__(self):
        self.position = np.linspace(-1.2, 0.6, 9)
        self.velocity = np.linspace(0, 0.07, 9)

    def transform(self, observation):
        position, velocity = observation
        return build_state([
            to_bin(position, self.position),
            to_bin(velocity, self.velocity)
        ])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        state_space_size = 10**env.observation_space.shape[0]
        action_space_size = env.action_space.n
        print(f"State Space Size: {state_space_size} | Action Space Size: {action_space_size}")
        self.q_table = np.random.uniform(low=-1, high=1, size=(state_space_size, action_space_size))
        print(f"Q-table created with shape {self.q_table.shape}")
        self.state_history = []

    def predict(self, state):
        x = self.feature_transformer.transform(state)
        return self.q_table[x]

    def take_action(self, state, epsilon):
            if np.random.random() > epsilon: #This is exploitation
                state = self.predict(state)
                return np.argmax(state) #Current state, max value
            else: #This is exploration
                return self.env.action_space.sample()

    def update(self, state, new_state, action, reward):
        discrete_state = self.feature_transformer.transform(state)
        discrete_new_state = self.feature_transformer.transform(new_state)
        if new_state[0] >= self.env.goal_position:
            self.q_table[discrete_state, action] = 0
        else:
            self.q_table[discrete_state, action] = (1 - LEARNING_RATE) * self.q_table[discrete_state, action] + LEARNING_RATE * (reward + GAMMA * np.max(self.q_table[discrete_new_state, :]))
            self.state_history.append(discrete_state)



def play_episode(model, epsilon, display):

    state = model.env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = model.take_action(state, epsilon)

        new_state, reward, done, info = model.env.step(action)
        total_reward += reward

        if display:
            model.env.render()

        if done and new_state[0] >= model.env.goal_position:
            reward = 100 #Tinker with this reward and you'll realize how often the state will be repeated (considering the reward is high)

        model.update(state, new_state, action, reward)
        steps += 1

        state = new_state


    return total_reward

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


if __name__ == '__main__':

    NUM_EPSIODES = 10_000
    epsilon = 1
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = NUM_EPSIODES // 2
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    display = False

    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)


    totalrewards = np.zeros(NUM_EPSIODES)
    for episode in range(NUM_EPSIODES):
        totalreward = play_episode(model, epsilon, display)
        totalrewards[episode] = totalreward

        display = False

        if episode % 100 == 0:
            print("Episode:", episode, "Total reward:", totalreward, "Epsilon:", epsilon)
            print("Average reward for last 100 episodes: " + "{0:.2f}".format(np.sum(totalrewards[episode - 100:episode]) / 100))
        
        if episode % 1000 == 0:
            display = True
            
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    plt.hist(model.state_history)
    plt.title('States')
    plt.show()
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
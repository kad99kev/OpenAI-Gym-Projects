# !pip install tensorflow-gpu

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

print(tf.__version__)

def build_policy_predict(ALPHA, n_actions, input_dims, layer1_size, layer2_size):
  inputs = tf.keras.Input(shape=(input_dims,))
  advantages = tf.keras.Input(shape=[1])

  dense1 = tf.keras.layers.Dense(layer1_size, input_dim=input_dims, activation='relu')(inputs)
  dense2 = tf.keras.layers.Dense(layer2_size, activation='relu')(dense1)
  output = tf.keras.layers.Dense(n_actions, activation='softmax')(dense2)

  def custom_loss(y_true, y_pred):
    out = tf.clip_by_value(y_pred, 1e-8, 1-1e-8)
    log_lik = y_true * tf.math.log(out)

    return tf.keras.backend.sum(-log_lik * advantages)

  policy = tf.keras.Model([inputs, advantages], [output])
  policy.compile(optimizer=tf.keras.optimizers.Adam(ALPHA), loss=custom_loss)

  predict = tf.keras.Model(inputs, [output])
  predict.compile(optimizer=tf.keras.optimizers.Adam(ALPHA), loss=custom_loss)

  return policy, predict

class Agent:
  def __init__(self, ALPHA, GAMMA=0.99, n_actions=4, 
               layer1_size=16, layer2_size=16, input_dims=128, 
               file_name='rModel.h5'):
    self.gamma = GAMMA
    self.lr = ALPHA
    self.discount = 0
    self.input_dims = input_dims
    self.actions = n_actions
    self.action_space = [i for i in range(n_actions)]
    self.state_memory = []
    self.action_memory = []
    self.reward_memory = []
    self.policy_model, self.predict_model = build_policy_predict(ALPHA, n_actions, input_dims, layer1_size, layer2_size)
    self.file_name = file_name


  def choose_action(self, observation):
    # Observation is the state itself
    state = observation[np.newaxis, :]
    probabilities = self.predict_model.predict(state)[0]
    # Make a choice based on probabilities
    action = np.random.choice(self.action_space, p=probabilities)
    return action

  def store_transition(self, observation, action, reward):
    self.action_memory.append(action)
    self.reward_memory.append(reward)
    self.state_memory.append(state)

  def learn(self):
    state_memory = np.array(self.state_memory)
    reward_memory = np.array(self.reward_memory)
    action_memory = np.array(self.action_memory)

    # np.arange gets each row and (, action memory) will get its respecitve column value
    actions = np.zeros([len(action_memory), self.actions])
    actions[np.arange(len(action_memory)), action_memory] = 1
    
    G = np.zeros_like(reward_memory)
    for t in range(len(reward_memory)):
      G_sum = 0
      discount = 1
      for k in range(t, len(reward_memory)):
        G_sum += reward_memory[k] * discount
        discount *= self.gamma
      G[t] = G_sum

    # Scaling and normalizing
    mean = np.mean(G)
    std = np.std(G) if np.std(G) > 0 else 1
    self.discount = (G-mean) / std 
    # G is discount


    cost = self.policy_model.train_on_batch([state_memory, self.discount], actions)

    # Resetting the memories after episode
    self.state_memory = []
    self.reward_memory = []
    self.action_memory = []

    return cost

  def save_model(self):
     self.policy_model.save_model(self.file_name)

  def load_model(self):
    self.policy_model = tf.keras.model.load_model(self.file_name)

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


if __name__ == '__main__':
  env = gym.make('LunarLander-v2')
  print(f"Observation space {env.observation_space.shape[0]}")
  print(f"Action space {env.action_space.n}")

  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n


  agent = Agent(ALPHA=0.0001, input_dims=state_size, GAMMA=0.99, n_actions=action_size,
                layer1_size=64, layer2_size=64)

  print(f"Summary of Policy Model\n {agent.policy_model.summary()}")
  print(f"Summary of Predict Model\n {agent.predict_model.summary()}")

  score_history = []

  NUM_EPISODES = 10_000


  for episode in range(NUM_EPISODES):
    done = False
    score = 0
    state = env.reset()
    steps = 0
    
    while not done:
      steps += 1
      action = agent.choose_action(state)
      new_state, reward, done, info = env.step(action)

      # print(f"State {new_state}")
      # print(f"Action {action}")
      
      agent.store_transition(state, action, reward)
      state = new_state
      score += reward
    
    score_history.append(score)
    agent.learn()

    if episode % 100 == 0:
      print(f"Total reward for Episode: {episode} is {score}")
      print(f"Average reward for last 100 episodes is {np.average(score_history[-100:])}")

  plot_running_avg(np.array(score_history))


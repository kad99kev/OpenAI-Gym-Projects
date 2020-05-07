import gym
import numpy as np
import matplotlib.pyplot as plt

from dqn import Brain

env = gym.make('LunarLander-v2')

BUFFER_SIZE = 64
GAMMA = 0.99

TOTAL_EPISODES = 500
EPS = 1.0
EPS_START = 1.0
EPS_END = TOTAL_EPISODES // 10
DECAY_RATE = 1/(EPS_END - EPS_START)

brain = Brain(8, 256, 4, BUFFER_SIZE, GAMMA, EPS, EPS_START, EPS_END, DECAY_RATE)

scores = []
eps_history = []


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


for episode in range(TOTAL_EPISODES):
    score = 0
    eps_history.append(brain.EPS['EPS'])
    state = env.reset()
    done = False
    while not done:
        action = brain.select_action(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        if done: next_state = None
        brain.store_transitions(state, action, next_state, reward)
        brain.optimize_model()
        state = next_state
    brain.update_eps(episode)
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print(f"Episode: {episode} | Score: {score} | Average Score: {avg_score} | Epsilon: {brain.EPS['EPS']}")

x = [i+1 for i in range(TOTAL_EPISODES)]
plot_learning_curve(x, scores, eps_history, filename='lander_run1_rms.png')
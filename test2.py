import gym
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    for _ in xrange(200):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    best_params = None
    best_reward = 0

    search_length = 10000
    total_search = 2000
    reach_list = [0 for _ in range(300)]

    for _ in range(total_search):
        for i in xrange(10000):
            parameters = np.random.rand(4) * 2 - 1
            reward = run_episode(env, parameters)
            if reward >= best_reward:
                best_reward = reward
                best_params = parameters
                if reward == 200:
                    reach_list[i] += 1
                    break;


    add_reach_list = [0 for _ in range(300)]
    for i in range(len(reach_list)):
        add_reach_list[i] = i * reach_list[i]
    print sum(add_reach_list) / float(total_search)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(reach_list)
    fig.savefig('reach_list.png')

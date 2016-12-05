import gym

def run_episode(env, )

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.reset()
    for _ in range(100):
        #env.render()
        observation, reward, done, info = env.step(0)

        print env.action_space

        print observation

        if done == True:
            env.reset()
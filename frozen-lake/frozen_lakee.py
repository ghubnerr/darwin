import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=True):
    epsilon = 1.0
    epsilon_decay_rate = 1e-5
    learning_rate = 0.99
    gamma = 0.99
    min_epsilon = 0.01

    env = gym.make('FrozenLake-v1',desc=None, map_name="8x8", is_slippery = False,render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()
        print(q)
    
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state, _ = env.reset()
        done = False

        print(f"{q=}")
        print(f"{epsilon=}")
        print(f"{i=}")
        while (not done):
            if is_training and rng.random() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            q[state, action] = (1 - learning_rate) * q[state, action] + (learning_rate) * (reward + gamma * np.max(q[new_state, :]))

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)
        rewards_per_episode[i] = reward
    
    env.close()
    
    if(is_training):
        f = open('frozen_lake8x8.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

if __name__ == "__main__":
    run(10000, is_training=True, render=False)
    run(100, is_training=False, render=True)
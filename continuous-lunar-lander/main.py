from ddpg import Agent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train():
    env = gym.make(
        "LunarLander-v2",
        continuous = True,
        gravity = -10.0,
        render_mode = None
    )

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=4)

    np.random.seed(0)
    score_history = []

    for i in range(1000):
        done = False
        score = 0
        obs, _ = env.reset()
        while not done:
            print(obs.shape)
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
        print("episode", i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]))
        if i % 25 == 0:
            agent.save_models()

    plt.plot(score_history)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

def load_trained():
    env = gym.make(
        "LunarLanderContinuous-v2",
        render_mode = "human"
    )

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=4)
    agent.load_models()

    np.random.seed(0)
    score_history = []

    for i in range(50):
        done = False
        score = 0
        obs, _ = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            score += reward
            obs = new_state

        score_history.append(score)
        print("episode", i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]))


    plt.plot(score_history)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

load_trained()
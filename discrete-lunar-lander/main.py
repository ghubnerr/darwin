import gymnasium as gym
import numpy as np
from dqn import Agent, DeepQNetwork
import pickle

def learn_to_land():
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[8], n_actions=4, max_mem_size=1000000, batch_size=64, eps_end=0.01, eps_dec=0.99)

    scores, eps_history = [], []
    n_games = 500

    for _ in range(n_games):
        score = 0
        done = False
        observation, info = env.reset(seed=42)
        while not done:   
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            score += reward

            done = terminated or truncated

            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        
        print('episode ', _, ' score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

    env.close()
    with open('trained_agent.pkl', 'wb') as f:
        pickle.dump((agent.Q_eval, agent.epsilon, scores, eps_history), f)

def land():
    env = gym.make("LunarLander-v2", render_mode="human")
    with open('trained_agent.pkl', 'rb') as f:
        Q_eval, epsilon, scores, eps_history = pickle.load(f)
        agent = Agent(gamma=0.99, epsilon=epsilon, lr=0.003, input_dims=[8], n_actions=4, max_mem_size=1000000, batch_size=64, eps_end=0.01, eps_dec=0.99, Q_eval=Q_eval)
        
        n_games = 500
        scores, eps_history = [], []

        for _ in range(n_games):
            score = 0
            done = False
            observation, info = env.reset(seed=42)
            
            while not done:   
                action = agent.choose_action(observation)
                observation_, reward, terminated, truncated, info = env.step(action)
                score += reward

                done = terminated or truncated

                observation = observation_

            scores.append(score)
            eps_history.append(agent.epsilon)

            avg_score = np.mean(scores[-100:])
            
            print('episode ', _, ' score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        env.close()


if __name__ == "__main__":
   #learn_to_land()
    land()
# Project DARWIN -- Diving Into Reinforcement Learning
Visit Individual Folders for Demo Videos! 
![Main Screen](./atari/assets/main.png)

- Project Inspired by OpenAI's "Emergent Tool Use from Multi-Agent Autocurricula" [Link Here](https://github.com/openai/multi-agent-emergence-environments)
  
## RL in a Nutshell

![Atari training](https://github.com/ghubnerr/darwin/assets/91924667/ec58d1b9-e8c4-4822-b7d7-60bc3739052e)

## Example Training Loop 
```python
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
```
### The Team


<img width="868" alt="image" src="https://github.com/ghubnerr/darwin/assets/91924667/ab4ceeb9-465e-4c67-8938-f837e117de87">

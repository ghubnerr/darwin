Reinforcement Learning for Multi-Agent Game Environments using Maskable Proximal Policy Optimization
-

Installation Requirements:
-
To work on this project, the following dependencies are required.
- pettingzoo[classic]>=1.24.0
- stable-baselines3>=2.0.0
- sb3-contrib>=2.0.0

Observations:
- 
- The agents would play a long, unstructured game until one of the players won. Even after training for 2,500,000 steps, this behavior was not corrected. This lead to the belief that agents are rewarded solely for winning the game and not for making "smart" moves.
- The reward function perhaps was not punishing the agent for missing obvious winning plays.
- The Maskable PPO architecture's only goal appears to be ensuring that agents do not make illegal action moves in the game of Connect 4.
- The algorithm is designed only to check for an illegal move, to improve performance of the AI, the model will have to be modified to include better reward functions for making unwise moves.

Training and Evaluation:
- 
This model uses Stable-Baseline3 to train agents in the Connect 4 environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Note: You can comment out the line related to training the agent for 100 games against a random agent. That is only a test to see how to game works, and adds no value to the training process.
Do not train for more than 2,000,000 steps, as the algorithm is not incentivized to improve beyond that or less.

## Installations

You can make the necessary installations with the following code:

```
pip install gymnasium numpy matplotlib
pip install gymnasium[toy-text]
```

## Results

https://github.com/ghubnerr/darwin/assets/110702967/b8d9d229-98a5-4e04-ad1a-e3b37aa17412

## Usage

The 'cliffwalking.py' script can be run directly from the command line. It will train the agent for a specified number of episodes and then test the learned policy.
The agent's Q-table will be saved to 'cliffwalking.pkl' after training, and this file will be loaded for testing the policy.

## Features

- Implements Q-learning algorithm for the CliffWalking-v0 environment.
- Supports both training and evaluation modes.

## Configuration 

You can adjust the training parameters at the beginning of the run function in the script:
- episodes: Number of episodes to run the agent.
- is_training: Set to True to train the agent or False to evaluate.
- render: Set to True to render the environment (only in evaluation mode).

<h1>HALF CHEETAH DDPG MODEL</h1>

Reinforcement Learning for Continuous Action Spaces and Continuous Observation Spaces

https://github.com/ghubnerr/darwin/assets/35817749/39f2ab8f-323d-48e9-8489-32ac3b2d5724

<h2>Preliminary Information</h2>
<h3>Required Packages</h3>
mujoco
gym
matplotlib (for plotting learn history)

<h3>Action Space</h3>
TL;DR: There are 6 continuous action spaces.

The action space is a Box(-1, 1, (6,), float32). An action represents the torques applied at the hinge joints.

<h3>Observation Space</h3>
TL;DR: There are 17 continuous observable spaces.

Observations consist of positional values of different body parts of the cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

By default, observations do not include the cheetah’s rootx. It may be included by passing exclude_current_positions_from_observation=False during construction. In that case, the observation space will be a Box(-Inf, Inf, (18,), float64) where the first element represents the rootx. Regardless of whether exclude_current_positions_from_observation was set to true or false, the will be returned in info with key "x_position".

<h3>Rewards</h3>
The reward consists of two parts:

forward_reward: A reward of moving forward which is measured as forward_reward_weight * (x-coordinate before action - x-coordinate after action)/dt. dt is the time between actions and is dependent on the frame_skip parameter (fixed to 5), where the frametime is 0.01 - making the default dt = 5 * 0.01 = 0.05. This reward would be positive if the cheetah runs forward (right).

ctrl_cost: A cost for penalising the cheetah if it takes actions that are too large. It is measured as ctrl_cost_weight * sum(action2) where ctrl_cost_weight is a parameter set for the control and has a default value of 0.1

The total reward returned is reward = forward_reward - ctrl_cost and info will also contain the individual reward terms

<h2>Results</h2>

For the current model shown above, the model was left to train with the following parameters:
Alpha: 0.000025
Beta: 0.000025
Episodes: 1500

This model took approximately 6 hours to train.
![ep1500alphabeta000025](https://github.com/ghubnerr/darwin/assets/35817749/bd17923f-d2c5-45f0-8755-325ffd587e2b)

<h3>Train() and load_trained()</h3>
load_trained() function loads a pre-trained model that ran through 1000 episodes of training, while train() does training from scratch. You can edit which one of the functions is running from the bottom of the main.py file. If you set render_mode=False, the program will train a lot faster.







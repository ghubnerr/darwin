# The Humanoid-v4 Environment 

## The Raw Model


https://github.com/ghubnerr/darwin/assets/91924667/b0f5b13a-a823-4575-b6bc-0fbb72ce7763


## Our Trained Model (Using Deep Deterministic Policy Gradients)

https://github.com/ghubnerr/darwin/assets/91924667/93a6db8a-2927-4863-ab42-a49fd80e5720

## Action Space
The action space is a Box(-1, 1, (17,), float32). An action represents the torques applied at the hinge joints.

## Observation Space
Observations consist of positional values of different body parts of the Humanoid, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

## Rewards
The reward consists of three parts:

`healthy_reward` Every timestep that the humanoid is alive (see section Episode Termination for definition), it gets a reward of fixed value healthy_reward

`forward_reward`: A reward of walking forward which is measured as forward_reward_weight * (average center of mass before action - average center of mass after action)/dt. dt is the time between actions and is dependent on the frame_skip parameter (default is 5), where the frametime is 0.003 - making the default dt = 5 * 0.003 = 0.015. This reward would be positive if the humanoid walks forward (in positive x-direction). The calculation for the center of mass is defined in the .py file for the Humanoid.

`ctrl_cost`: A negative reward for penalising the humanoid if it has too large of a control force. If there are nu actuators/controls, then the control has shape nu x 1. It is measured as ctrl_cost_weight * sum(control2).

`contact_cost`: A negative reward for penalising the humanoid if the external contact force is too large. It is calculated by clipping contact_cost_weight * sum(external contact force2) to the interval specified by contact_cost_range.

The total reward returned is `reward = healthy_reward + forward_reward - ctrl_cost - contact_cost` and info will also contain the individual reward terms

## Requirements 
On Python 3.10.12:
```bash
pip install gymnasium
pip install gymnasium[mujoco]
```

## Getting started
From the terminal, run:
```
python main.py
```

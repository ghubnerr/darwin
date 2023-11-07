https://github.com/ghubnerr/darwin/assets/91924667/cf4e724d-817d-4c70-9c7c-04f3650b4604

The Gymnasium Lunar Lander environment is a classic rocket trajectory optimization problem. It simulates the task of landing a rocket on the moon while efficiently managing fuel consumption. Here's a summary of its key attributes:

Action Space:

The action space is discrete and consists of four possible actions:
0. Do nothing
Fire the left orientation engine
Fire the main engine
Fire the right orientation engine
Observation Space:

The observation space is an 8-dimensional vector:
The coordinates of the lander in the x and y dimensions
Linear velocities in the x and y dimensions
The angle of the lander
Angular velocity
Two booleans indicating whether each leg of the lander is in contact with the ground or not.
Description:

The goal is to land the rocket safely on the moon's surface, with the landing pad always located at coordinates (0, 0).
There are two versions of the environment: discrete and continuous.
Fuel is infinite, so the agent can learn to fly and land on its first attempt.
Rewards:

The rewards are given after each step, and the total episode reward is the sum of rewards across all steps in the episode.
Rewards are affected by various factors:
Reward increases as the lander gets closer to the landing pad and decreases as it moves further away.
Reward increases for slower lander movement and decreases for faster movement.
Reward decreases as the lander tilts away from a horizontal orientation.
Reward increases by 10 points for each leg in contact with the ground.
Reward decreases when the side engine fires (-0.03 points per frame) and when the main engine fires (-0.3 points per frame).
The episode can receive an additional reward of -100 or +100 points for crashing or landing safely, respectively.
A successful episode is one in which the agent scores at least 200 points.
Starting State:

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.
Episode Termination:

The episode terminates under several conditions, including:
The lander crashes (contact with the moon's surface).
The lander moves outside of the viewport (x coordinate greater than 1).
The lander is not awake (i.e., it doesn't move or collide with any other body).
The Lunar Lander environment provides an interesting challenge for reinforcement learning agents to learn efficient rocket landing strategies. It involves making decisions about when to fire the engine and in which direction to achieve a safe landing.

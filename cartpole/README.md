Proximal Policy Optimization (PPO) on Discrete Action Space Environments
-

Untrained Preview
-

Trained Preview
-
https://github.com/ghubnerr/darwin/assets/91924667/1403a870-6fa2-45a8-bdf5-5746dd6f4f92

Observations
-
- In the trained model, the agent was able to successfully maintain the balance of the sitck/pole with minimal angular discrepancies. 
- The agent is able to main an upright position of the stick until the episode terminates


Dependencies
-
The Environment: 
```
import gymnasium as gym
gym.make('CartPole-v1')
```


Description
-
This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in “Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem”. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

This environment specifications and documentation can be found on Open AI Gym's website: 
https://gymnasium.farama.org/environments/classic_control/cart_pole/

Action Space
-

The action is a ndarray with shape (1,) which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.

0: Push cart to the left

1: Push cart to the right

Note: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

Rewards
-
A reward of +1 is given for every step taken, including the termination step, if the agent is able to maintain an upright position of the stick for as long as possible. The threshold for rewards is 500 for v1.

Sources
-

Open AI Gymnasium: https://gymnasium.farama.org/environments/classic_control/cart_pole/

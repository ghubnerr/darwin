# Continuous Lunar Lander Environment with DDPG
Reinforcement Learning for Continuous Action Spaces and Continuous Observation Spaces 



https://github.com/ghubnerr/continuous_lunar_lander/assets/91924667/961fc1da-3fc0-47b7-9fac-c073d2354cd5



## Observations
- [X] The agent had much more angular control, but struggled to pivot to the sides, as it may have been rewarded more for staying upright than to head to the land zone
- [X] The reward function was probably not punishing the agent enough for landing outside of the landing zone.
- [X] Rather, the agent preferred to do a successful landing (touch its legs on the floor), than to position itself correctly in the zone.
- [X] Continuous control (**DDPG**) gave it a better grip and the movement was not so loose as its [Discrete version](https://github.com/ghubnerr/lunar_lander).

## Continuous Action Space
If ```continuous=True``` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the action space will be ```Box(-1, +1, (2,), dtype=np.float32)```. The first coordinate of an action determines the throttle of the main engine, while the second coordinate specifies the throttle of the lateral boosters. Given an action ```np.array([main, lateral])```, the main engine will be turned off completely if ```main < 0``` and the throttle scales affinely from 50% to 100% for ```0 <= main <= 1``` (in particular, the main engine doesn’t work with less than 50% power). Similarly, if ```-0.5 < lateral < 0.5```, the lateral boosters will not fire at all. If ```lateral < -0.5```, the left booster will fire, and if ```lateral > 0.5```, the right booster will fire. Again, the throttle scales affinely from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).
Documentation: Gymnasium
## Usage and Packages
```pip install torch gymnasium 'gymnasium[box2d]'```

You might need to install Box2D Separately, which requires a swig package to compile code from Python into C/C++, which is the language that Box2d was built in:

```brew install swig```

```pip install box2d```

## Average Score: 164.38 (significant improvement from discrete action spaces)
For each step, the reward:

- is increased/decreased the closer/further the lander is to the landing pad.
- is increased/decreased the slower/faster the lander is moving.
- is decreased the more the lander is tilted (angle not horizontal).
- is increased by 10 points for each leg that is in contact with the ground.
- is decreased by 0.03 points each frame a side engine is firing.
- is decreased by 0.3 points each frame the main engine is firing.
The episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively. An episode is considered a solution if it scores at least 200 points.**

## ```train()``` and ```load_trained()```
```load_trained()``` function loads a pre-trained model that ran through 1000 episodes of training, while ```train()``` does training from scratch. You can edit which one of the functions is running from the bottom of the main.py file. If you set render_mode=False, the program will train a lot faster.

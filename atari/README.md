# Darwin Atari Project

![Main Screen](./assets/main.png)

GUI tool to train Gynasium atari games

**Select an atari game to train, change some hyperparameters, and watch the model train!**

# Features

Selection of [Hyperparameters](https://towardsdatascience.com/artificial-intelligence-hyperparameters-48fa29daa516) lke learning rate, batch size, device to train on, etc

Caches Neural Network, records videos, and logs the performance of the model over time

# Examples of outputs

## [Breakout](https://gymnasium.farama.org/environments/atari/breakout/)

### Short Description

You move a paddle and hit the ball in a brick wall at the top of the screen. Your goal is to destroy the brick wall. You can try to break through the wall and let the ball wreak havoc on the other side, all on its own! You have five lives.

Action Space: One of 4 Discrete actions

Observation Space: 210 x 160 x 3 array

| Epoch | Video                                  | Model                                                  |
| ----- | -------------------------------------- | ------------------------------------------------------ |
| 2045  | ![](./assets/breakout/videos/2045.mp4) | [Breakout 2045.pth](./assets/breakout/models/2045.pth) |
| 2000  | ![](./assets/breakout/videos/200.mp4)  | [2000.pth](./assets/breakout/models/2000.pth)          |
| 1900  | ![](./assets/breakout/videos/1900.mp4) | [1900.pth](./assets/breakout/models/1900.pth)          |
| 1800  | ![](./assets/breakout/videos/1800.mp4) | [1800.pth](./assets/breakout/models/1800.pth)          |
| 1700  | ![](./assets/breakout/videos/1700.mp4) | [1700.pth](./assets/breakout/models/1700.pth)          |
| 1600  | ![](./assets/breakout/videos/1600.mp4) | [1600.pth](./assets/breakout/models/1600.pth)          |
| 1500  | ![](./assets/breakout/videos/1500.mp4) | [1500.pth](./assets/breakout/models/1500.pth)          |
| 1400  | ![](./assets/breakout/videos/1400.mp4) | [1400.pth](./assets/breakout/models/1400.pth)          |
| 1300  | ![](./assets/breakout/videos/1300.mp4) | [1300.pth](./assets/breakout/models/1300.pth)          |
| 1200  | ![](./assets/breakout/videos/1200.mp4) | [1200.pth](./assets/breakout/models/1200.pth)          |
| 1100  | ![](./assets/breakout/videos/1100.mp4) | [1100.pth](./assets/breakout/models/1100.pth)          |
| 1000  | ![](./assets/breakout/videos/1000.mp4) | [1000.pth](./assets/breakout/models/1000.pth)          |
| 900   | ![](./assets/breakout/videos/900.mp4)  | [900.pth](./assets/breakout/models/900.pth)            |
| 800   | ![](./assets/breakout/videos/800.mp4)  | [800.pth](./assets/breakout/models/800.pth)            |
| 700   | ![](./assets/breakout/videos/700.mp4)  | [700.pth](./assets/breakout/models/700.pth)            |
| 600   | ![](./assets/breakout/videos/600.mp4)  | [600.pth](./assets/breakout/models/600.pth)            |
| 500   | ![](./assets/breakout/videos/500.mp4)  | [500.pth](./assets/breakout/models/500.pth)            |
| 400   | ![](./assets/breakout/videos/400.mp4)  | [400.pth](./assets/breakout/models/400.pth)            |
| 300   | ![](./assets/breakout/videos/300.mp4)  | [300.pth](./assets/breakout/models/300.pth)            |
| 200   | ![](./assets/breakout/videos/200.mp4)  | [200.pth](./assets/breakout/models/200.pth)            |
| 100   | ![](./assets/breakout/videos/100.mp4)  | [100.pth](./assets/breakout/models/100.pth)            |
| 0     | ![](./assets/breakout/videos/0.mp4)    | [0.pth](./assets/breakout/models/0.pth)                |

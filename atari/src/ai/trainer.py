import json
import math
import os
import random
from collections import deque
from datetime import datetime
from itertools import count
from typing import List, Tuple

import matplotlib as mpl

backend_ = mpl.get_backend()
mpl.use("Agg")  # Prevent showing stuff

import gymnasium as gym
import matplotlib.pyplot as plt
import nptyping as npt
import pandas as pd
import seaborn as sns
import torch
import torch as T
import torch.nn as nn
from numpy import Infinity
from shortuuid import uuid
from src.ai.model import DQN, DuelDQN
from src.ai.utils import ReplayMemory, Transition, VideoRecorder
from src.ai.wrapper import AtariWrapper
from src.gameList import GameDict
from torch import optim

obs_type = npt.NDArray[npt.Shape["84, 84"], npt.Number]

AVG_OVER = 25


class Trainer:
    GAMMA = 0.99  # bellman function
    EPS_START = 1
    EPS_END = 0.05
    EPS_DECAY = 50000
    WARMUP = 1000  # don't update net until WARMUP steps
    MEMORY_SIZE = 30_000

    steps_done = 0
    warmupstep = 0
    n_epochs = 0
    eps_threshold = EPS_START

    env: gym.Env[obs_type, int]

    n_action: int  # number of actions for the env

    policy_net: nn.Module
    target_net: nn.Module
    optimizer: optim.Optimizer
    memory: ReplayMemory

    log_dir: str
    video: VideoRecorder
    obs: obs_type

    rewards: List[float] = []
    losses: List[float] = []

    reward_queue = deque(maxlen=AVG_OVER)
    losses_queue = deque(maxlen=AVG_OVER)

    avg_rewards: List[float] = []
    avg_losses: List[float] = []
    saved = False

    def __init__(
        self,
        game: GameDict,
        model="dqn",
        device="cpu",
        lr=2.5e-4,
        epochs=10001,
        batch_size=32,
        use_ddqn=False,
        eval_freq=500,
        max_timesteps=2_500,
        max_timesteps_calc="lowest",
        logging=False,
        data_path="./data",
        **kwargs: float,
    ) -> None:
        self.game = game
        self.env_name = game["env"]
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_ddqn = use_ddqn
        self.eval_freq = eval_freq
        self.data_path = data_path

        self.logging = logging

        for k, v in kwargs.items():
            setattr(self, k.upper(), v)

        self.env = gym.make(self.env_name)
        self.spec = self.env.spec
        default_max_timesteps = (
            self.spec.max_episode_steps
            if self.spec.max_episode_steps != None
            else math.inf
        )

        if max_timesteps_calc == "lowest":
            self.spec.max_episode_steps = min(max_timesteps, default_max_timesteps)
        elif max_timesteps_calc == "highest":
            self.spec.max_episode_steps = max(max_timesteps, default_max_timesteps)
        elif max_timesteps_calc == "override":
            self.spec.max_episode_steps = max_timesteps

        self.env = AtariWrapper(self.env, max_episode_steps=self.spec.max_episode_steps)
        self.n_action = self.env.action_space.n  # type: ignore

        if use_ddqn:
            methodname = f"double_{model}"
        else:
            methodname = model

        self.uuid = uuid()[:10]
        self.id = f"{self.env_name.split('/')[-1]}-{methodname}-{self.uuid}"

        self.log_dir = os.path.join(self.data_path, self.id)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.video_log_dir = os.path.join(self.log_dir, "videos")
        self.model_dir = os.path.join(self.log_dir, "models")

        if not os.path.exists(self.video_log_dir):
            os.makedirs(self.video_log_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.video = VideoRecorder(self.video_log_dir)

        if model == "dqn":
            self.policy_net = DQN(in_channels=4, n_actions=self.n_action).to(device)
            self.target_net = DQN(in_channels=4, n_actions=self.n_action).to(device)
        else:
            self.policy_net = DuelDQN(in_channels=4, n_actions=self.n_action).to(device)
            self.target_net = DuelDQN(in_channels=4, n_actions=self.n_action).to(device)
        # let target model = model
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(self.MEMORY_SIZE)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    def warm_up(self) -> None:
        """
        Warm up the memory with random actions

        Recommended to run before training
        """
        if self.logging:
            print("Warming up...")

        for _epoch in count():
            self.warm_up_epoch()

            if self.finished_warmup():
                break

    def finished_warmup(self) -> bool:
        return self.warmupstep > self.WARMUP

    def warm_up_epoch(self):
        self.obs, _info = self.env.reset()  # (84,84)
        self.obs = torch.from_numpy(self.obs).to(self.device)  # type: ignore # (84,84)
        # stack four frames together, hoping to learn temporal info
        self.obs = torch.stack((self.obs, self.obs, self.obs, self.obs)).unsqueeze(
            0
        )  # type: ignore # (1,4,84,84)

        for _step in count():
            done = self.warm_up_step()

            if done:
                break

    def warm_up_step(self):
        # take one step
        action = torch.tensor([[self.env.action_space.sample()]]).to(self.device)
        next_obs, reward, terminated, truncated, _info = self.env.step(
            action.item()  # type: ignore
        )
        done = terminated or truncated

        # convert to tensor
        reward = torch.tensor([reward], device=self.device)  # (1)
        done = torch.tensor([done], device=self.device)  # (1)
        next_obs = torch.from_numpy(next_obs).to(self.device)  # (84,84)
        next_obs = torch.stack(
            (next_obs, self.obs[0][0], self.obs[0][1], self.obs[0][2])
        ).unsqueeze(
            0
        )  # (1,4,84,84)

        # store the transition in memory
        self.memory.push(self.obs, action, next_obs, reward, done)

        # move to next state
        self.obs = next_obs  # type: ignore

        self.warmupstep += 1

        return done

    def epoch(self) -> Tuple[float, float]:
        """Does one epoch of training

        Returns total reward and total loss for the epoch
        """

        self.obs, _info = self.env.reset()  # (84,84)
        self.obs = torch.from_numpy(self.obs).to(self.device)  # type: ignore # (84,84)
        # stack four frames together, hoping to learn temporal info
        self.obs = torch.stack((self.obs, self.obs, self.obs, self.obs)).unsqueeze(0)  # type: ignore # (1,4,84,84)

        total_loss = 0.0
        total_reward = 0

        done = False
        while not done:
            reward, loss, done = self.step()

            total_reward += reward
            total_loss += loss

        self.n_epochs += 1

        self.rewards.append(total_reward)
        self.losses.append(total_loss)

        self.reward_queue.append(total_reward)
        self.losses_queue.append(total_loss)

        avg_reward = sum(self.reward_queue) / len(self.reward_queue)
        avg_loss = sum(self.losses_queue) / len(self.losses_queue)

        self.avg_rewards.append(avg_reward)
        self.avg_losses.append(avg_loss)

        return total_reward, total_loss

    def step(
        self,
    ) -> Tuple[float, float, bool]:
        """Does a single step of an epoch

        Returns reward and loss for the step
        """
        # take one step
        action = self.select_action(self.obs)  # type: ignore
        next_obs, reward, terminated, truncated, _info = self.env.step(action.item())  # type: ignore
        done = terminated or truncated
        n_done = done

        i_reward = reward

        # convert to tensor
        reward = torch.tensor([reward], device=self.device)  # (1)
        done = torch.tensor([done], device=self.device)  # (1)
        next_obs = torch.from_numpy(next_obs).to(self.device)  # (84,84)
        next_obs = torch.stack(
            (next_obs, self.obs[0][0], self.obs[0][1], self.obs[0][2])
        ).unsqueeze(
            0
        )  # (1,4,84,84)

        # store the transition in memory
        self.memory.push(self.obs, action, next_obs, reward, done)

        # move to next state
        self.obs = next_obs  # type: ignore

        # train
        self.policy_net.train()
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(
            *zip(*transitions)
        )  # batch-array of Transitions -> Transition of batch-arrays.
        state_batch = torch.cat(batch.state)  # (bs,4,84,84)
        next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
        action_batch = torch.cat(batch.action)  # (bs,1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
        done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)

        # Q(st,a)
        state_qvalues = self.policy_net(state_batch)  # (bs,n_actions)
        selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)

        with torch.no_grad():
            # Q'(st+1,a)
            next_state_target_qvalues = self.target_net(
                next_state_batch
            )  # (bs,n_actions)
            if self.use_ddqn:
                # Q(st+1,a)
                next_state_qvalues = self.policy_net(next_state_batch)  # (bs,n_actions)
                # argmax Q(st+1,a)
                next_state_selected_action = next_state_qvalues.max(1, keepdim=True)[
                    1
                ]  # (bs,1)
                # Q'(st+1,argmax_a Q(st+1,a))
                next_state_selected_qvalue = next_state_target_qvalues.gather(
                    1, next_state_selected_action
                )  # (bs,1)
            else:
                # max_a Q'(st+1,a)
                next_state_selected_qvalue = next_state_target_qvalues.max(
                    1, keepdim=True
                )[
                    0
                ]  # (bs,1)

        # td target
        tdtarget = (
            next_state_selected_qvalue * self.GAMMA * ~done_batch + reward_batch
        )  # (bs,1)

        # optimize
        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_state_qvalue, tdtarget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # let self.target_net = policy_net every 1000 steps
        if self.steps_done % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.logging and done and self.n_epochs % self.eval_freq == 0:
            self.log()

        return i_reward, loss.item(), n_done  # type: ignore

    def log(self):
        """Logs the performance of the model, records a video and saves the model

        If logging is set to true this is called every 500 epochs by default
        """
        with torch.no_grad():
            self.video.reset()
            evalenv = gym.make(self.env_name)
            evalenv = AtariWrapper(evalenv, video=self.video)
            self.obs, info = evalenv.reset()  # type: ignore
            self.obs = torch.from_numpy(self.obs).to(self.device)  # type: ignore
            self.obs = torch.stack((self.obs, self.obs, self.obs, self.obs)).unsqueeze(
                0
            )  # type: ignore
            evalreward = 0
            self.policy_net.eval()
            for _ in count():
                action = self.policy_net(self.obs).max(1)[1]
                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = evalenv.step(action.item())
                evalreward += reward  # type: ignore
                next_obs = torch.from_numpy(next_obs).to(self.device)  # (84,84)
                next_obs = torch.stack(
                    (next_obs, self.obs[0][0], self.obs[0][1], self.obs[0][2])
                ).unsqueeze(
                    0
                )  # (1,4,84,84)
                self.obs = next_obs  # type: ignore
                if terminated or truncated:
                    if (
                        info["lives"] == 0 or _ >= self.spec.max_episode_steps
                    ):  # real end
                        break
                    else:
                        self.obs, info = evalenv.reset()  # type: ignore
                        self.obs = torch.from_numpy(self.obs).to(self.device)  # type: ignore
                        self.obs = torch.stack(
                            (self.obs, self.obs, self.obs, self.obs)
                        ).unsqueeze(
                            0
                        )  # type: ignore
            evalenv.close()
            self.video.save(f"{self.n_epochs}.mp4")

            self.save()
            print(f"Eval epoch {self.n_epochs}: Reward {evalreward}")

    def save(self, dir: str | None = None) -> bool:
        """Saves the model and other data to a file

        Returns if the saved
        """

        dir = dir if dir is not None else self.model_dir

        if self.n_epochs != 0 and len(self.rewards) == 0:
            return

        try:
            torch.save(
                self.policy_net,
                os.path.join(dir, f"{self.n_epochs}.pth"),
            )

            self.save_metadata()
            self.save_graph()

            return True
        except Exception as e:
            return False

    def save_metadata(self):
        d = os.path.join(self.log_dir, "metadata.json")
        data = {
            **self.game,
            "rewards": self.rewards,
            "losses": self.losses,
            "avg_rewards": self.avg_rewards,
            "avg_losses": self.avg_losses,
            "id": self.id,
            "env": self.env_name,
            "lr": self.lr,
            "steps": self.steps_done,
            "batch_size": self.batch_size,
            "use_ddqn": self.use_ddqn,
            "eval_freq": self.eval_freq,
            "device": self.device,
            "epochs": self.n_epochs,
            "steps": self.steps_done,
            "created": datetime.now().isoformat(),
        }

        json.dump(data, open(d, "w"), indent=2)

    def save_graph(self):
        if len(self.rewards) == 0:
            return

        d = os.path.join(self.log_dir, "performance.png")
        data = []
        for i, (r, l, ar, al) in enumerate(
            zip(self.rewards, self.losses, self.avg_rewards, self.avg_losses)
        ):
            data += [(i, "Reward", r)]
            data += [(i, "Loss", l)]
            data += [(i, "Avg Reward", ar)]
            data += [(i, "Avg Loss", al)]

        frame = pd.DataFrame(
            data=data,
            columns=["epoch", "type", "value"],
        )

        sns.set_theme(style="darkgrid")
        sns.lineplot(data=frame, x="epoch", y="value", hue="type")
        plt.savefig(d)
        plt.close()

    def select_action(self, state: T.Tensor) -> T.Tensor:
        """
        epsilon greedy
        - epsilon: choose random action
        - 1-epsilon: argmax Q(a,s)

        Input: state shape (1,4,84,84)

        Output: action shape (1,1)
        """
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]]).to(self.device)

    def close(self):
        """Closes the environment and frees up resources

        You should call this once training is done to free up resources
        But make sure to save before because this will delete the model and any other data
        """

        try:
            self.env.close()
            del self.env
            del self.policy_net
            del self.target_net
            del self.optimizer
            del self.memory
            del self.video
        except:
            pass

    def save_and_close(self):
        """Saves and closes the environment"""

        self.log()
        self.close()

    def training_loop(
        self,
    ) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
        """Runs the full training loop

        Returns (rewardList, lossList), (avgrewardlist, avglosslist)
        """
        self.warm_up()

        if self.logging:
            print("Training...")

        rewardList = []
        lossList = []
        rewarddeq = deque([], maxlen=100)
        lossdeq = deque([], maxlen=100)
        avgrewardlist = []
        avglosslist = []

        for n_epoch in range(self.epochs):
            total_reward, total_loss = self.epoch()

            rewardList.append(total_reward)
            lossList.append(total_loss)
            rewarddeq.append(total_reward)
            lossdeq.append(total_loss)
            avgreward = sum(rewarddeq) / len(rewarddeq)
            avgloss = sum(lossdeq) / len(lossdeq)
            avglosslist.append(avgloss)
            avgrewardlist.append(avgreward)

            if self.logging:
                print(
                    f"Epoch {n_epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {self.eps_threshold:.2f}, TotalStep {self.steps_done}"
                )

        self.env.close()

        if self.logging != None:
            # plot loss-epoch and reward-epoch
            plt.figure(1)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.plot(range(len(lossList)), lossList, label="loss")
            plt.plot(range(len(lossList)), avglosslist, label="avg")
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, "loss.png"))

            plt.figure(2)
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            plt.plot(range(len(rewardList)), rewardList, label="reward")
            plt.plot(range(len(rewardList)), avgrewardlist, label="avg")
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, "reward.png"))

        return (rewardList, lossList), (avgrewardlist, avglosslist)

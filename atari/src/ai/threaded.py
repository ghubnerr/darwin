import threading
import time
from contextlib import nullcontext
from datetime import timedelta
from itertools import count
from typing import Callable, List

import torch as T
from IPython.core.magics.execution import _format_time
from src.ai.trainer import Trainer, obs_type
from src.gameList import GameDict
from src.ui.util import the_void
from ui.settings import get_setting

from .utils import VideoRecorder


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.

    From stack overflow -> https://stackoverflow.com/a/325528
    """

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def end(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ThreadedTrainer(StoppableThread):
    trainer: Trainer

    def __init__(
        self,
        game: GameDict,
        on_epoch: Callable[[List[float], List[float], List[float], int], None],
        on_update: Callable[[str], None],
        on_done: Callable[[], None],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.on_epoch = on_epoch
        self.on_done = on_done
        self.on_update = on_update

        trainer_args = {
            "lr": get_setting("lr"),
            "epochs": get_setting("epochs"),
            "batch_size": get_setting("batch_size"),
            "use_ddqn": get_setting("use_ddqn"),
            "eval_freq": get_setting("eval_freq"),
            "device": get_setting("device"),
            "max_timesteps": get_setting("max_timesteps"),
            "max_timesteps_calc": get_setting("max_timesteps_calc"),
            "data_path": get_setting("data_path"),
        }

        self.trainer = Trainer(
            game,
            *args,
            **trainer_args,
            **kwargs,
        )

    def run(self):
        if self.check_stopped():
            return

        self.on_update("Warming up")
        for _epoch in count():
            self.trainer.warm_up_epoch()

            if self.check_stopped():
                return

            if self.trainer.finished_warmup():
                break

        self.on_update("Warm up done... Starting training")

        if self.check_stopped():
            return

        start_time = time.monotonic()
        for n_epoch in range(self.trainer.epochs):
            if self.check_stopped():
                return

            total_reward, total_loss = self.trainer.epoch()
            end_time = time.monotonic()
            delta = timedelta(seconds=end_time - start_time)

            self.on_update(
                f"Epoch #{n_epoch} -> R:[b]{total_reward:.2f}[/b] L:[b]{total_loss:.2f}[/b] T:{_format_time(delta.microseconds / 100_000)}"
            )

            self.on_epoch(
                self.trainer.rewards,
                self.trainer.losses,
                self.trainer.avg_rewards,
                self.trainer.n_epochs,
            )
            start_time = time.monotonic()

        self.trainer.save_and_close()
        self.on_done()

    def check_stopped(self):
        if self.stopped():
            self.on_end()

            return True

        return False

    def on_end(self):
        self.on_update("Stopping training")
        self.on_done()
        self.trainer.save_and_close()


class ThreadedEvaluator(ThreadedTrainer):
    def __init__(
        self,
        game: GameDict,
        on_frame: Callable[[obs_type], None],
        trained_model: str,
        *args,
        **kwargs,
    ):
        self.on_frame = on_frame
        self.trained_model = trained_model

        super().__init__(
            game,
            the_void,
            the_void,
            the_void,
            trained_model=trained_model,
            video=FrameRecorder(on_frame, "___pain___"),
            *args,
            **kwargs,
        )

    def run(self):
        if self.check_stopped():
            return

        with T.no_grad():
            while True:
                if self.check_stopped():
                    return

                self.trainer.eval_epoch()


class FrameRecorder(VideoRecorder):
    def __init__(self, on_frame: Callable[[obs_type], None], *args, **kwargs):
        self.on_frame = on_frame

        super().__init__(*args, **kwargs)

    def record(self, frame):
        self.on_frame(frame)
        return super().record(frame)

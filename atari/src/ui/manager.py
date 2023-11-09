import multiprocessing
import os
from typing import List

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.label import Label
from kivy_garden.graph import Graph, LinePlot, MeshLinePlot, Plot, SmoothLinePlot
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screenmanager import ScreenManager
from src.ai.train import ThreadedTrainer
from src.ai.utils import clamp
from src.gameList import GameDict, gameList
from src.ui.game import Game
from src.ui.gameModal import GameModal

KV = """
<Manager>:
    id: manager

    MDScreen:
        name: "main"

        MDAnchorLayout:
            anchor_x: "right"
            anchor_y: "top"
            padding_top: 50

            MDRectangleFlatIconButton:
                icon: "cog"
                text: "Settings"
                on_press: app.open_settings()

        MDAnchorLayout:
            anchor_x: "center"
            anchor_y: "top"
            padding: 0, 100, 0, 0

            MDScrollView:
                MDGridLayout:
                    padding: 10, 50
                    cols: 5
                    id: images_grid
                    size_hint_y: None
                    height: self.minimum_height  #<<<<<<<<<<<<<<<<<<<<
                    spacing: 10
                    row_default_height: "300dp"
                    col_default_width: "200dp"
                    col_force_default: True


    MDScreen:
        name: "progress"
        id: progress_screen

        MDBoxLayout:
            halign: "center"
            orientation: "vertical"
            id: box

            MDRectangleFlatIconButton:
                icon: "cancel"
                text: "Cancel Training"
                pos_hint: {"center_x": 0.5}
                on_press: root.stop_training()

            MDLabel:
                font_size: "30sp"
                id: training_txt
                text: root.training_text
                markup: True
                pos_hint: {"center_x": 0.5}
                size_hint_y: 0.5

"""

Builder.load_string(KV)


class Manager(ScreenManager):
    dialog: MDDialog = None  # type: ignore
    training_text = StringProperty("")

    queue: multiprocessing.Queue

    trainer: ThreadedTrainer
    training: bool = False

    graph: Graph
    reward_plt: Plot
    loss_plt: Plot
    avg_plt: Plot

    rewards: List[float] = []
    losses: List[float] = []

    training_txt_widget: Label

    def __init__(self, **kwargs):
        Clock.schedule_once(self.on_start)

        super().__init__(**kwargs)

    def stop_training(self):
        if self.training:
            self.trainer.stop()

    def check_if_trained(self, data: GameDict) -> bool:
        directory = "./models"
        all_folders = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]
        env_slug = f"log_{data['slug']}"
        return env_slug in all_folders

    def on_game_press(self, data: GameDict, *_, **__):
        env_id = data["env"]

        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None  # type: ignore

        app = MDApp.get_running_app()
        self.dialog = MDDialog(
            title=data["name"],
            height="500dp",
            type="custom",
            content_cls=GameModal(data),
            buttons=[
                MDFlatButton(
                    text="Cancel",
                    theme_text_color="Custom",
                    text_color=app.theme_cls.secondary_text_color,
                    on_press=lambda *_, **__: self.dialog.dismiss(),
                ),
                MDFlatButton(
                    text="Start Training",
                    theme_text_color="Custom",
                    text_color=app.theme_cls.primary_color,
                    on_press=lambda *_, **__: self.begin_training(env_id),
                ),
            ],
        )
        self.dialog.open()

    def begin_training(self, env_id: str, *_, **__):
        self.dialog.dismiss()
        self.current = "progress"

        if self.training:
            return

        self.trainer = ThreadedTrainer(
            env_id,
            on_epoch=self.on_epoch,
            on_done=self.on_done,
            on_update=self.on_update,
        )
        self.trainer.start()
        self.training = True

    def on_epoch(self, total_reward: float, total_loss: float):
        length = len(self.reward_plt.points)
        self.reward_plt.points.append((length - 1, total_reward))
        self.loss_plt.points.append((length - 1, total_loss))

        self.rewards.append(total_reward)
        self.losses.append(total_loss)

        m = length - 25 if length > 25 else length
        avg = sum(self.rewards[m:]) / min(25, max(length, 1))
        self.avg_plt.points.append((length - 1, avg))

        highest = max(*self.rewards, *self.losses[1:], 1)
        lowest = min(*self.rewards, *self.losses[1:], 0)

        self.graph.ymax = int(highest)
        self.graph.ymin = int(lowest - 1)

        if length > 100:
            self.graph.xmin = length - 100
            self.graph.xmax = length

    def on_done(self):
        print("Done Training")

        self.trainer = None

        self.training = False
        self.rewards = []
        self.losses = []

        self.reward_plt.points = []
        self.loss_plt.points = []
        self.avg_plt.points = []

        self.current = "main"

    def on_update(self, txt: str):
        self.training_txt_widget.text = txt

    def on_start(self, *_, **__):
        grid = self.ids["images_grid"]

        for game in gameList.values():
            g = Game(game, on_press=self.on_game_press)
            g.on_press = self.on_game_press
            grid.add_widget(g)

        self.reward_plt = SmoothLinePlot(color=[0, 1, 0, 1])
        self.loss_plt = SmoothLinePlot(color=[1, 0, 0, 1])
        self.avg_plt = SmoothLinePlot(color=[0, 0, 1, 1])

        self.graph = Graph(
            xlabel="Epoch",
            ylabel="Value",
            x_ticks_minor=5,
            x_ticks_major=100,
            x_grid_label=True,
            x_grid=True,
            y_ticks_major=1,
            y_ticks_minor=1,
            y_grid_label=True,
            y_grid=True,
            padding=5,
            xmin=0,
            xmax=100,
            ymin=0,
            ymax=1,
            size_hint=(0.9, 0.9),
        )

        self.graph.add_plot(self.reward_plt)
        self.graph.add_plot(self.loss_plt)
        self.graph.add_plot(self.avg_plt)

        self.ids["box"].add_widget(self.graph)

        self.training_txt_widget = self.ids["training_txt"]

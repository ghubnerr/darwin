from typing import List

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import DictProperty, StringProperty
from kivy_garden.graph import Graph, Plot, SmoothLinePlot
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from src.ai.threaded import ThreadedTrainer
from src.gameList import GameDict
from src.ui.util import Util, go_to_screen_callback, kivy_callback

KV = """
<TrainingScreen>:
    name: "progress"
    id: progress_screen

    MDBoxLayout:
        halign: "center"
        orientation: "vertical"
        id: box

        MDRectangleFlatIconButton:
            icon: "cancel"
            text: "Cancel Training"
            pos_hint: {"center_x": 0.8}
            on_press: root.stop_training()

        MDLabel:
            font_size: "40sp"
            id: title
            text: f"[b]{root.title}[/b]"
            markup: True
            pos_hint: {"center_x": 0.5}
            size_hint_y: 0.3
            halign: "center"

        MDLabel:
            font_size: "30sp"
            id: training_txt
            text: root.training_text
            markup: True
            pos_hint: {"center_x": 0.5}
            size_hint_y: 0.25
            halign: "center"
"""
Builder.load_string(KV)


class TrainingScreen(MDScreen, Util):
    training_text = StringProperty("")
    data: GameDict = None
    title = StringProperty("")

    trainer: ThreadedTrainer
    training: bool = False

    graph: Graph
    reward_plt: Plot
    loss_plt: Plot
    avg_plt: Plot

    trained_games: List[GameDict]

    rewards: List[float] = []
    losses: List[float] = []

    training_txt_widget: MDLabel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_init(self, *_, **__):
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

    @kivy_callback
    def begin_training(self, data: GameDict, *args):
        # If it is already training, ignore!
        if self.training:
            return

        self.data = data
        self.title = data["name"]

        self.trainer = ThreadedTrainer(
            data,
            on_epoch=self.on_epoch,
            on_done=self.on_done,
            on_update=self.on_update,
        )
        self.trainer.start()
        self.training = True

    def on_epoch(
        self,
        rewards: List[float],
        losses: List[float],
        avg_list: List[float],
        _epoch: int,
    ):
        length = len(self.rewards)
        self.reward_plt.points.append((length - 1, rewards[-1]))
        self.loss_plt.points.append((length - 1, losses[-1]))

        self.rewards = rewards
        self.losses = losses

        self.avg_plt.points.append((length - 1, avg_list[-1]))

        highest = max(*rewards, *losses[1:], 1)
        lowest = min(*rewards, *losses[1:], 0)

        self.graph.ymax = int(highest)
        self.graph.ymin = int(lowest - 1)

        if length > 100:
            self.graph.xmin = length - 100
            self.graph.xmax = length

    def on_done(self):
        print("Done Training")

        self.training = False
        self.rewards = []
        self.losses = []

        self.reward_plt.points = []
        self.loss_plt.points = []
        self.avg_plt.points = []

        Clock.schedule_once(go_to_screen_callback("main"))

    def stop_training(self):
        if self.training:
            self.training = False
            self.trainer.end()

    def on_update(self, txt: str):
        self.training_txt_widget.text = txt

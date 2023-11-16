from itertools import chain
from os import path

from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from src.load_trained import TrainedGame
from src.ui.components import LoadTrainedClickable
from src.ui.util import Util, kivy_callback

KV = """
<LoadTrained>:
    name: "trained"

    MDScrollView:
        MDGridLayout:
            padding: 10, 50
            cols: 5
            id: models
            size_hint_y: None
            height: self.minimum_height  #<<<<<<<<<<<<<<<<<<<<
            spacing: 10
            row_default_height: "300dp"
            col_default_width: "200dp"
            col_force_default: True

"""

Builder.load_string(KV)


class LoadTrained(MDScreen, Util):
    @kivy_callback
    def load_trained(self, data: TrainedGame):
        models = self.ids["models"]

        epochs = [int(path.basename(f).split(".")[0]) for f in data["models"]]

        # epochs = set(range(0, data["epochs"], data["eval_freq"]))
        # epochs.add(data["epochs"])
        for epoch in sorted(epochs):
            clickable = LoadTrainedClickable(data, epoch)
            clickable.on_release = self.on_clickable_press
            models.add_widget(clickable)

    def on_clickable_press(self, epoch: int):
        print(epoch)

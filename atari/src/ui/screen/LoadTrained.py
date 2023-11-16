from itertools import chain
from os import path

from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from src.load_trained import TrainedGame
from src.ui.components import LoadTrainedClickable
from src.ui.util import Util, events, go_to_screen, kivy_callback

KV = """
<LoadTrained>:
    name: "trained"


    MDScrollView:
        MDGridLayout:
            padding: 10, 100
            cols: 5
            id: models
            size_hint_y: None
            height: self.minimum_height  #<<<<<<<<<<<<<<<<<<<<
            spacing: 10
            row_default_height: "300dp"
            col_default_width: "200dp"
            col_force_default: True

    MDFloatLayout:
        MDRectangleFlatIconButton:
            size_hint: None, None
            pos_hint: {"top": 1, "center_x": 0.5}
            icon: "arrow-left"
            text: "Go back"
            on_press: root.go_back()

"""

Builder.load_string(KV)


@events("load_trained_at_epoch")
class LoadTrained(MDScreen, Util):
    data: TrainedGame

    @kivy_callback
    def load_trained(self, data: TrainedGame):
        models = self.ids["models"]
        self.data = data

        epochs = [int(path.basename(f).split(".")[0]) for f in data["models"]]
        models.clear_widgets()

        # epochs = set(range(0, data["epochs"], data["eval_freq"]))
        # epochs.add(data["epochs"])
        for epoch in sorted(epochs, reverse=True):
            clickable = LoadTrainedClickable(data, epoch)
            clickable.on_release = self.on_clickable_press
            models.add_widget(clickable)

    def on_clickable_press(self, epoch: int):
        self.dispatch(
            "on_load_trained_at_epoch",
            (
                self.data,
                epoch,
            ),
        )

    def go_back(self, *_):
        go_to_screen("main")

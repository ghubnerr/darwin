from os import path
from typing import Callable

from kivy.lang import Builder
from kivy.properties import StringProperty
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from src.load_trained import TrainedGame
from src.ui.settings import get_setting

from .Link import Link

KV = """
<LoadGameModalContent>:
    orientation: "vertical"
    size_hint_y: None
    height: "400dp"
    pos_hint: {"center_x": 0.5, "center_y": 0.5}
    spacing: 20


    MDBoxLayout:
        orientation: "horizontal"
        Image:
            id: image
            source: root.source
            size_hint: None, None
            allow_stretch: True
            keep_ratio: True
            width: "300dp"
            height: "200dp"

        MDLabel:
            text: root.prop
            font_size: "20dp"

    MDBoxLayout:
        orientation: "horizontal"
        spacing: 10

        Link:
            more_text: "Env Docs: "
            url: root.url

"""

Builder.load_string(KV)


class LoadGameModalContent(MDBoxLayout):
    source = StringProperty("")
    name = StringProperty("")
    url = StringProperty("")
    prop = StringProperty("")

    INCLUDED_PROPERTIES = [
        "epochs",
        "steps",
        "lr",
        "batch_size",
        "use_ddqn",
        "eval_freq",
        "device",
    ]

    def __init__(self, data: TrainedGame, **kwargs):
        self.data = data

        self.prop = "Trained with: \n" + " \n".join(
            [f"{s.capitalize()}: {data[s]}" for s in self.INCLUDED_PROPERTIES]
        )

        self.name = data["name"]
        self.url = data["url"]
        self.source = path.abspath(
            path.join(get_setting("data_path"), data["id"], "performance.png")
        )

        super().__init__(**kwargs)


class LoadGameModal(MDDialog):
    def __init__(
        self, data: TrainedGame, start_training: Callable[[TrainedGame], None], **kwargs
    ):
        self.data = data
        self.start_training = start_training
        self.app = MDApp.get_running_app()

        super().__init__(
            title=data["name"],
            height="500dp",
            type="custom",
            content_cls=LoadGameModalContent(data),
            buttons=[
                MDFlatButton(
                    text="Cancel",
                    theme_text_color="Custom",
                    text_color=self.app.theme_cls.secondary_text_color,
                    on_press=lambda *_: self.dismiss(),
                ),
                MDFlatButton(
                    text="View Models & Videos",
                    theme_text_color="Custom",
                    text_color=self.app.theme_cls.primary_color,
                    on_press=lambda *_,: self.training_callback(),
                ),
            ],
            **kwargs,
        )

    def training_callback(self):
        self.dismiss()
        self.start_training(self.data)

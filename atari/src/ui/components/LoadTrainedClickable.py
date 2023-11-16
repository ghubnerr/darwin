from os import path

from kivy.lang import Builder
from kivy.properties import DictProperty, NumericProperty, StringProperty
from kivy.uix.video import Video
from kivymd.uix.boxlayout import MDBoxLayout
from src.load_trained import TrainedGame
from src.ui.components.Game import (
    ButtonBehavior,
    RectangularRippleBehavior,
    SusSmartTile,
)

KV = """
<LoadTrainedClickable>:
    lines: 2

    ClickableBox:
        id: image
        size_hint: None, None
        size_hint_y: 1 if root.overlap else None
        height: root.height if root.overlap else root.height - box.height
        on_release: root.dispatch("on_release", root.epoch)
        on_press: root.dispatch("on_press", root.epoch)
        width: root.width
        height: root.height

        Video:
            id: image
            source: root.source
            size_hint_y: 1 if root.overlap else None
            height: root.height if root.overlap else root.height - box.height
            width: root.width
            height: root.height
            state: 'play'
            options: root.options
            preview: root.preview

    AtariOverlayBox:
        id: box
        md_bg_color: root.box_color
        size_hint_y: None
        padding: "8dp"
        radius: root.box_radius
        height: "68dp" if root.lines == 2 else "48dp"
        pos:
            (0, 0) \
            if root.box_position == "footer" else \
            (0, root.height - self.height)

        MDLabel:
            text: f"Epoch #{root.epoch}"

            bold: True
            color: 1, 1, 1, 1
            width: "300dp"
"""
Builder.load_string(KV)


class ClickableBox(RectangularRippleBehavior, ButtonBehavior, MDBoxLayout):
    pass


class LoadTrainedClickable(SusSmartTile):
    data: TrainedGame
    epoch = NumericProperty(0)
    options = DictProperty({"eos": "loop", "speed": 0.25})
    preview = StringProperty("")

    def __init__(self, data: TrainedGame, epoch: int, *args, **kwargs):
        self.data = data
        self.epoch = epoch

        self.source = path.join(data["path"], "videos", f"{epoch}.mp4")
        self.preview = f"src/assets/atari/{self.data['slug']}.gif.png"

        super().__init__(*args, **kwargs)

    def add_widget(self, widget, *args, **kwargs):
        return super().add_widget(widget, *args, **kwargs)

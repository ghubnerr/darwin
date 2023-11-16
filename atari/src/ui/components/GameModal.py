import webbrowser

from gameList import GameDict
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivymd.uix.boxlayout import MDBoxLayout

KV = """
<GameModal>:
    orientation: "horizontal"
    size_hint_y: None
    height: "400dp"
    pos_hint: {"center_x": 0.5, "center_y": 0.5}
    spacing: 20

    Image:
        size_hint: None, None
        size_hint_y: 1
        allow_stretch: True
        keep_ratio: True
        source: root.img
        width: "200dp"
        height: "300dp"

    MDBoxLayout:
        orientation: "vertical"
        spacing: 10

        MDLabel:
            text: root.description
            font_size: "20dp"

        MDLabel:
            markup: True
            text: f"[u][ref=link]{root.url}[/ref][/u]"
            font_size: "20dp"
            color: (0, 0, 1, 1)
            on_ref_press: root.on_ref_press()

"""

Builder.load_string(KV)


class GameModal(MDBoxLayout):
    img = StringProperty("")
    name = StringProperty("")
    description = StringProperty("")
    url = StringProperty("")

    def __init__(self, data: GameDict, **kwargs):
        self.data = data

        self.img = f"src/assets/atari/{data['slug']}.gif.png"
        self.name = data["name"]
        self.description = data["description"]
        self.url = data["url"]

        super().__init__(**kwargs)

    def on_ref_press(self, *args):
        webbrowser.open(self.url)

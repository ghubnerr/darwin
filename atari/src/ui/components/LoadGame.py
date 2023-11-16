from humanize import naturaldate
from kivy.lang import Builder
from kivy.properties import StringProperty
from src.load_trained import Game
from src.ui.components.Game import SusSmartTile

KV = """
<LoadGame>:
    lines: 2

    AtariImage:
        id: image
        size_hint: None, None
        allow_stretch: True
        keep_ratio: True
        mipmap: root.mipmap
        source: root.source
        radius: root.radius if root.radius else [0, 0, 0, 0]
        size_hint_y: 1 if root.overlap else None
        height: root.height if root.overlap else root.height - box.height
        pos:
            ((0, 0) if root.overlap else (0, box.height)) \
            if root.box_position == "footer" else \
            (0, 0)
        on_release: root.dispatch("on_release", root.data)
        on_press: root.dispatch("on_press", root.data)
        width: root.width
        height: root.height

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
            text: f"{root.name} - {root.epochs} Created: {root.created}"

            bold: True
            color: 1, 1, 1, 1
            width: "300dp"
"""
Builder.load_string(KV)


class LoadGame(SusSmartTile):
    name = StringProperty("")
    created = StringProperty("")
    epochs = StringProperty("")
    # properties = StringProperty("")
    data: Game

    def __init__(self, data: Game, *args, **kwargs):
        self.data = data

        self.name = data["name"]
        self.epochs = str(data["epochs"])
        self.created = naturaldate(data["created"])

        # self.properties = " \n".join(
        #     [f"{s.capitalize()}: {data[s]}" for s in self.INCLUDED_PROPERTIES]
        # )

        self.source = f"src/assets/atari/{self.data['slug']}.gif.png"

        super().__init__(*args, **kwargs)

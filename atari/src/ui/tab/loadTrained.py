from typing import List

from kivy.lang import Builder
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from src.load_trained import Game, load_trained
from src.ui.components import LoadGame, LoadGameModal
from src.ui.tab.util import Tab
from src.ui.util import Util

KV = """
<LoadTrainedTab>:
    title: "Load Trained"
    content_text: "Load the already trained AI"

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
"""

Builder.load_string(KV)


class LoadTrainedTab(Tab, Util):
    trained: List[Game]

    def post_init(self, *_, **__):
        self.trained = load_trained()
        grid = self.ids["images_grid"]

        for game in self.trained:
            g = LoadGame(game, on_press=self.on_game_press)
            g.on_press = self.on_game_press
            grid.add_widget(g)

    def on_game_press(self, data: Game, *_, **__):
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None

        self.dialog = MDDialog(
            title=data["name"],
            height="500dp",
            type="custom",
            content_cls=LoadGameModal(data),
            buttons=[
                MDFlatButton(
                    text="Cancel",
                    theme_text_color="Custom",
                    text_color=self.app.theme_cls.secondary_text_color,
                    on_press=lambda *_, **__: self.dialog.dismiss(),
                ),
                MDFlatButton(
                    text="Start Training",
                    theme_text_color="Custom",
                    text_color=self.app.theme_cls.primary_color,
                    on_press=lambda *_, **__: self.start_training(data),
                ),
            ],
        )
        self.dialog.open()

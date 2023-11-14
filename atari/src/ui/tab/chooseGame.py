import os

from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from src.gameList import GameDict, gameList
from src.ui.game import Game
from src.ui.gameModal import GameModal
from src.ui.tab.util import Tab
from src.ui.util import Util, events

KV = """
<ChooseGameTab>:
    title: "Training"
    content_text: "Train the Ai"

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


@events("begin_training")
class ChooseGameTab(Tab, Util):
    dialog: MDDialog = None

    def post_init(self, *_, **__):
        grid = self.ids["images_grid"]

        for game in gameList.values():
            g = Game(game, on_press=self.on_game_press)
            g.on_press = self.on_game_press
            grid.add_widget(g)

    def on_game_press(self, data: GameDict, *_, **__):
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None

        self.dialog = MDDialog(
            title=data["name"],
            height="500dp",
            type="custom",
            content_cls=GameModal(data),
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

    def start_training(self, data):
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None
            self.dispatch("on_begin_training", data)

    def check_if_trained(self, data: GameDict) -> bool:
        directory = "./models"
        all_folders = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]
        env_slug = f"log_{data['slug']}"
        return env_slug in all_folders

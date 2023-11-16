import src.ui.screen
import src.ui.screen.training
from kivy.lang import Builder
from kivymd.uix.screenmanager import ScreenManager

KV = """
<Manager>:
    id: manager

    MainScreen:
        id: main
        on_begin_training:
            root.current = "progress"
            training.begin_training(*args)
        on_load_game:
            root.current = "trained"
            trained.load_trained(*args)


    TrainingScreen:
        id: training
        on_done_training:
            main.reload_trained_tab()
            root.current = "main"

    LoadTrained:
        id: trained
"""

Builder.load_string(KV)


class Manager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_load_game(self, *args):
        print(*args)

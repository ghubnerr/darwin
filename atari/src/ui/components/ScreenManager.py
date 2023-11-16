import src.ui.screen
import src.ui.screen.training
from kivy.lang import Builder
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.screenmanager import ScreenManager
from kivymd.uix.tab import MDTabsBase


class Tab(MDFloatLayout, MDTabsBase):
    pass


KV = """
<Manager>:
    id: manager

    MainScreen:
        on_begin_training:
            root.current = "progress"
            training.begin_training(*args)
        on_load_game:
            root.current = "trained"
            trained.load_trained(*args)


    TrainingScreen:
        id: training

    LoadTrained:
        id: trained
"""

Builder.load_string(KV)


class Manager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_load_game(self, *args):
        print(*args)

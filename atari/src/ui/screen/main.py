from typing import List

import src.ui.tab.chooseGame
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy_garden.graph import Graph, Plot, SmoothLinePlot
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from src.ai.threaded import ThreadedTrainer
from src.gameList import GameDict
from src.ui.util import Util, events, go_to_screen_callback

KV = """
<MainScreen>:
    name: "main"

    MDTabs:
        ChooseGameTab:
            on_begin_training: root.dispatch("on_begin_training", *args)

    MDAnchorLayout:
        anchor_x: "right"
        anchor_y: "top"
        padding: 0, 10

        MDFillRoundFlatIconButton:
            icon: "cog"
            text: "Settings"
            on_press: app.open_settings()
"""

Builder.load_string(KV)


@events("begin_training")
class MainScreen(MDScreen):
    pass

from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from src.ui.tab import ChooseGameTab, LoadTrainedTab
from src.ui.util import events

KV = """
<MainScreen>:
    name: "main"

    MDTabs:
        ChooseGameTab:
            on_begin_training: root.dispatch("on_begin_training", *args)
        LoadTrainedTab:

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

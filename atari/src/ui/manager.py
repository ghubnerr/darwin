import json
import multiprocessing
import os
from datetime import datetime
from typing import Iterable, List, TypedDict

import src.ui.components
import src.ui.screen.main
import src.ui.screen.training
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy_garden.graph import Graph, Plot, SmoothLinePlot
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.screenmanager import ScreenManager
from kivymd.uix.tab import MDTabsBase
from src.ai.threaded import ThreadedTrainer
from src.gameList import GameDict, gameList
from src.load_trained import load_trained


class Tab(MDFloatLayout, MDTabsBase):
    pass


KV = """
<Manager>:
    id: manager

    MainScreen:
        on_begin_training:
            root.current = "progress"
            training.begin_training(*args)

    TrainingScreen:
        id: training

"""

Builder.load_string(KV)


class Manager(ScreenManager):
    pass

import json
from json import dumps, loads

import ui.manager
from kivy.lang import Builder
from kivy.uix.settings import SettingsWithSidebar
from kivymd.app import MDApp
from src.path import join

settings_list = loads(open(join("assets", "settings.json")).read())

# Filters out the cast property


def filter_out_cast_property(pair):
    key, value = pair
    if key == "cast":
        return False
    else:
        return True


settings = dumps(
    list(
        map(
            lambda d: dict(filter(filter_out_cast_property, d.items())),
            settings_list,
        )
    )
)

KV = """
# MDBoxLayout:
#     MDNavigationRail:
#         id: navigation_rail
#         md_bg_color: "#fffcf4"
#         selected_color_background: "#e7e4c0"
#         ripple_color_item: "#e7e4c0"
#         on_item_release: app.switch_screen(*args)
#             MDNavigationRailItem:
#                 text: "Train"
#                 icon: "robot-confused"

#             MDNavigationRailItem:
#                 text: "Load Model"
#                 icon: "robot-excited"
    # Manager:
Manager:
    id: manager

"""


class MainApp(MDApp):
    def build(self):
        self.settings_cls = SettingsWithSidebar
        self.use_kivy_settings = False

        return Builder.load_string(KV)

    def build_config(self, config):
        config.setdefaults(
            "AI Settings",
            {
                "lr": 2.5e-4,
                "epochs": 10001,
                "batch_size": "32",
                "use_ddqn": False,
                "eval_freq": 500,
                "max_timesteps": 2_500,
                "max_timesteps_calc": "lowest",
                "device": "cpu",
                "data_path": "./data",
            },
        )

    def build_settings(self, _settings):
        _settings.add_json_panel("AI Settings", self.config, data=settings)


MainApp().run()

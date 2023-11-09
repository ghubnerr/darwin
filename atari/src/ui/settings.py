from ast import literal_eval
from json import loads

from kivy.config import Config
from kivymd.app import MDApp

from src.path import join

settings = loads(open(join("assets", "settings.json")).read())


def get_setting(key: str):
    config: ConfigParser = MDApp.get_running_app().config  # type: ignore
    value = config.get("AI Settings", key)  # type: ignore

    matching_setting = [setting for setting in settings if setting["key"] == key][0]
    settings_type = matching_setting["type"]
    _cast = matching_setting["cast"] if "cast" in matching_setting else None

    match settings_type:
        case "numeric":
            if not _cast or _cast == "float":
                return float(value)
            else:
                return int(value)
        case "bool":
            return literal_eval(value)

        case _:
            return value

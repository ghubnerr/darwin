import webbrowser

from kivy.properties import StringProperty
from kivymd.uix.label import MDLabel
from src.ui.util import load_kv

KV = """
<Link>:
    markup: True
    text: f"{self.more_text}[u][ref=link]{self.url}[/ref][/u]"
    font_size: "20dp"
    color: (0, 0, 1, 1)
    on_ref_press: self.go_to()
"""
load_kv(KV)


class Link(MDLabel):
    url = StringProperty("")
    more_text = StringProperty("")

    def go_to(self, *_):
        webbrowser.open(self.url)

from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget


class TextureWidget(Widget):
    texture = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.texture = Texture.create(size=(250, 160))
        with self.canvas:
            self.rect = Rectangle(size=self.size, pos=self.pos, texture=self.texture)

    def update(self):
        self.canvas.ask_update()
        self.canvas.children[-1].texture = self.texture

from textwrap import shorten

from kivy.lang import Builder
from kivy.properties import (
    BooleanProperty,
    ColorProperty,
    ObjectProperty,
    OptionProperty,
    StringProperty,
    VariableListProperty,
)
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivymd import uix_path
from kivymd.theming import ThemableBehavior
from kivymd.uix.behaviors import RectangularRippleBehavior
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.fitimage import FitImage
from kivymd.uix.imagelist import MDSmartTile
from kivymd.uix.label import MDLabel
from kivymd.uix.relativelayout import MDRelativeLayout
from src.gameList import GameDict

KV = """
<Game>:
    lines: 2

    AtariImage:
        id: image
        size_hint: None, None
        allow_stretch: True
        keep_ratio: True
        mipmap: root.mipmap
        source: root.source
        radius: root.radius if root.radius else [0, 0, 0, 0]
        size_hint_y: 1 if root.overlap else None
        height: root.height if root.overlap else root.height - box.height
        pos:
            ((0, 0) if root.overlap else (0, box.height)) \
            if root.box_position == "footer" else \
            (0, 0)
        on_release: root.dispatch("on_release", root.data)
        on_press: root.dispatch("on_press", root.data)
        width: root.width
        height: root.height

    AtariOverlayBox:
        id: box
        md_bg_color: root.box_color
        size_hint_y: None
        padding: "8dp"
        radius: root.box_radius
        height: "68dp" if root.lines == 2 else "48dp"
        pos:
            (0, 0) \
            if root.box_position == "footer" else \
            (0, root.height - self.height)

        MDLabel:
            text: root.description

            bold: True
            color: 1, 1, 1, 1
            width: "300dp"
"""
Builder.load_string(KV)


class SusSmartTile(MDRelativeLayout, ThemableBehavior):
    """
    A tile for more complex needs.

    Includes an image, a container to place overlays and a box that can act
    as a header or a footer, as described in the Material Design specs.

    :Events:
        `on_press`
            Called when the button is pressed.
        `on_release`
            Called when the button is released (i.e. the touch/click that
            pressed the button goes away).
    """

    box_radius = VariableListProperty([0], length=4)
    """
    Box radius.

    .. versionadded:: 1.0.0

    .. code-block:: kv

        MDSmartTile:
            radius: 24
            box_radius: [0, 0, 24, 24]

    .. image:: https://github.com/HeaTTheatR/KivyMD-data/raw/master/gallery/kivymddoc/md-smart-tile-box-radius.png
        :align: center

    :attr:`box_radius` is an :class:`~kivy.properties.VariableListProperty`
    and defaults to `[0, 0, 0, 0]`.
    """

    box_color = ColorProperty((0, 0, 0, 0.5))
    """
    Sets the color in (r, g, b, a) or string format and opacity for the
    information box.

    .. code-block:: kv

        MDSmartTile:
            radius: 24
            box_radius: [0, 0, 24, 24]
            box_color: 0, 1, 0, .5

    .. image:: https://github.com/HeaTTheatR/KivyMD-data/raw/master/gallery/kivymddoc/md-smart-tile-box-color.png
        :align: center

    :attr:`box_color` is a :class:`~kivy.properties.ColorProperty`
    and defaults to `(0, 0, 0, 0.5)`.
    """

    box_position = OptionProperty("footer", options=["footer", "header"])
    """
    Determines weather the information box acts as a header or footer to the
    image. Available are options: `'footer'`, `'header'`.

    .. code-block:: kv

        MDSmartTile:
            radius: 24
            box_radius: [24, 24, 0, 0]
            box_position: "header"

    .. image:: https://github.com/HeaTTheatR/KivyMD-data/raw/master/gallery/kivymddoc/md-smart-tile-box-position.png
        :align: center

    :attr:`box_position` is a :class:`~kivy.properties.OptionProperty`
    and defaults to `'footer'`.
    """

    overlap = BooleanProperty(True)
    """
    Determines if the `header/footer` overlaps on top of the image or not.

    .. code-block:: kv

        MDSmartTile:
            radius: [24, 24, 0, 0]
            box_radius: [0, 0, 24, 24]
            overlap: False

    .. image:: https://github.com/HeaTTheatR/KivyMD-data/raw/master/gallery/kivymddoc/md-smart-tile-overlap.png
        :align: center

    :attr:`overlap` is a :class:`~kivy.properties.BooleanProperty`
    and defaults to `True`.
    """

    lines = OptionProperty(1, options=[1, 2])
    """
    Number of lines in the `header/footer`. As per `Material Design specs`,
    only 1 and 2 are valid values. Available are options: `1`, `2`.
    This parameter just increases the height of the container for custom
    elements.

    .. code-block:: kv

        MDSmartTile:
            radius: 24
            box_radius: [0, 0, 24, 24]
            lines: 2
            source: "cats.jpg"
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint: None, None
            size: "320dp", "320dp"

            MDIconButton:
                icon: "heart-outline"
                theme_icon_color: "Custom"
                icon_color: 1, 0, 0, 1
                pos_hint: {"center_y": .5}
                on_release: self.icon = "heart" if self.icon == "heart-outline" else "heart-outline"

            TwoLineListItem:
                text: "[color=#ffffff][b]My cats[/b][/color]"
                secondary_text: "[color=#808080][b]Julia and Julie[/b][/color]"
                pos_hint: {"center_y": .5}
                _no_ripple_effect: True

    .. image:: https://github.com/HeaTTheatR/KivyMD-data/raw/master/gallery/kivymddoc/md-smart-tile-lines.png
        :align: center

    :attr:`lines` is a :class:`~kivy.properties.OptionProperty`
    and defaults to `1`.
    """

    source = StringProperty()
    """
    Path to tile image. See :attr:`~kivy.uix.image.Image.source`.

    :attr:`source` is a :class:`~kivy.properties.StringProperty`
    and defaults to `''`.
    """

    mipmap = BooleanProperty(False)
    """
    Indicate if you want OpenGL mipmapping to be applied to the texture.
    Read :ref:`mipmap` for more information.

    .. versionadded:: 1.0.0

    :attr:`mipmap` is a :class:`~kivy.properties.BooleanProperty`
    and defaults to `False`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_event_type("on_release")
        self.register_event_type("on_press")

    def on_release(self, *args):
        """
        Called when the button is released (i.e. the touch/click that
        pressed the button goes away).
        """

    def on_press(self, *args):
        """Called when the button is pressed."""


class AtariImage(RectangularRippleBehavior, ButtonBehavior, Image):
    """Implements the tile image."""


class AtariOverlayBox(MDBoxLayout):
    """Implements a container for custom widgets to be added to the tile."""


class Game(SusSmartTile):
    game_id = StringProperty("")
    description = StringProperty("")
    name = StringProperty("")
    data = ObjectProperty()

    def __init__(self, data: GameDict, *args, **kwargs):
        self.data = data
        self.name = data["name"]

        self.source = f"src/assets/atari/{self.data['slug']}.gif.png"

        self.description = f"{self.name}: {shorten(self.data['description'], width=47, placeholder='...')}"

        super().__init__(*args, **kwargs)

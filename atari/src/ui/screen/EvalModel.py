from os import path
from typing import Tuple

from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.image import Image
from kivymd.uix.screen import MDScreen
from src.ai.threaded import ThreadedEvaluator
from src.ai.trainer import obs_type
from src.load_trained import TrainedGame
from src.ui.util import Util, do_later, go_to_screen, kivy_callback

KV = """
<EvalModel>:
    name: "eval"

    MDFloatLayout:
        MDRectangleFlatIconButton:
            size_hint: None, None
            pos_hint: {"top": 1, "center_x": 0.5}
            icon: "close-circle"
            text: "Stop Evaluating Model"
            on_press: root.stop()

    # TextureWidget:
    #     size: 250, 160
    #     id: game


"""

Builder.load_string(KV)


class EvalModel(MDScreen, Util):
    data: TrainedGame
    game: Image
    texture: Texture

    def post_init(self, *_, **__):
        # self.game = self.ids["game"]
        self.texture = Texture.create(size=(250, 160))

    @kivy_callback
    def load_trained_at_epoch(self, data: Tuple[TrainedGame, int]):
        self.data, epoch = data
        self.title = self.data["name"]

        model = path.join(self.data["path"], "models", f"{epoch}.pth")

        self.trainer = ThreadedEvaluator(
            self.data,
            on_frame=self.on_frame,
            trained_model=model,
        )
        self.trainer.start()
        self.training = True

    @do_later
    def on_frame(
        self,
        frame: obs_type,
    ):
        # do some kivy stuff
        # frame = frame.copy()
        # frame.resize((250, 160, 3), refcheck=False)
        # frame.flatten()
        # buf = frame.tobytes()
        # self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

        size = 250 * 160 * 3
        buf = [int(x * 255 / size) for x in range(size)]

        # then, convert the array to a ubyte string
        buf = bytes(buf)

        self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
        with self.canvas:
            Rectangle(texture=self.texture, pos=(0, 0), size=(250, 160))

        self.canvas.ask_update()

    def stop(self, *_):
        go_to_screen("main")

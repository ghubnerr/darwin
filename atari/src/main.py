# this is needed for supporting Windows 10 with OpenGL < v2.0
# Example: VirtualBox w/ OpenGL v1.1
import os
import platform

if platform.system() == "Windows":
    os.environ["KIVY_GL_BACKEND"] = "angle_sdl2"

import kivy
from src.ui.app import MainApp

kivy.require("1.0.6")  # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label

if __name__ == "__main__":
    MainApp().run()

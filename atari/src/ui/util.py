import os
from functools import wraps
from inspect import Parameter, signature
from typing import Callable

from kivy.clock import Clock
from kivy.lang import Builder
from kivymd.app import MDApp
from src.load_trained import TrainedGame
from src.ui.settings import get_setting


def go_to_screen(name: str):
    MDApp.get_running_app().root.current = name  # type: ignore


def go_to_screen_callback(name: str, *_, **__):
    return lambda *_, **__: go_to_screen(name)


class Util(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.app = MDApp.get_running_app()

        Clock.schedule_once(self.post_init)

    def post_init(self, *_, **__):
        pass


def events(*events: str):
    """Class decorator which takes in a list of events to register as `on_{name}`

    By default the event will bubble up to the main class
    """

    def inner(cls: object):
        orig_init = cls.__init__
        # Make copy of original __init__, so we can call it without recursion

        def __init__(self, *args, **kws):
            def event_handler(self, *_, **__):
                return False

            for event in events:
                bind(self, event_handler, as_name=f"on_{event}")

            orig_init(self, *args, **kws)  # type: ignore # Call the original __init__

            for event in events:
                self.register_event_type(f"on_{event}")

        cls.__init__ = __init__  # type: ignore # Set the class' __init__ to the new one
        return cls

    return inner


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def kivy_callback(func: Callable):
    def inner(*args, **kwargs):
        return func(args[0], args[-1])

    return inner


def load_kv(s: str) -> None:
    Builder.load_string(s)


def instance_variables(f):
    sig = signature(f)

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        values = sig.bind(self, *args, **kwargs)
        for k, p in sig.parameters.items():
            if k != "self":
                if k in values.arguments:
                    val = values.arguments[k]
                    if p.kind in (
                        Parameter.POSITIONAL_OR_KEYWORD,
                        Parameter.KEYWORD_ONLY,
                    ):
                        setattr(self, k, val)
                    elif p.kind == Parameter.VAR_KEYWORD:
                        for k, v in values.arguments[k].items():
                            setattr(self, k, v)
                else:
                    setattr(self, k, p.default)

    return wrapper


def do_later(f):
    """Wrapper function that schedules a function to be called on the next frame"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        Clock.schedule_once(lambda *_: f(*args, **kwargs), 0)

    return wrapper


def the_void(*args, **kwargs):
    pass

from typing import Callable, Iterable

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivymd.app import MDApp


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
    def inner(cls: object):
        orig_init = cls.__init__
        # Make copy of original __init__, so we can call it without recursion

        def __init__(self, *args, **kws):
            def event_handler(self, *_, **__):
                pass

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
        return func(args[0], args[-1], *args, **kwargs)

    return inner


def load_kv(s: str) -> None:
    Builder.load_string(s)

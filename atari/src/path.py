import os

path = os.path.dirname(os.path.abspath(__file__))


def join(*args: str):
    return os.path.join(path, *args)

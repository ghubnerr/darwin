import json
import os
from datetime import datetime
from os import path
from typing import Iterable, List, Tuple, TypedDict

from kivymd.app import MDApp
from src.ui.settings import get_setting


class Game(TypedDict):
    rewards: List[float]
    losses: List[float]
    avg_rewards: List[float]
    slug: str
    id: str
    env: str
    lr: float
    epochs: int
    batch_size: int
    use_ddqn: bool
    eval_freq: int
    device: str
    steps: int
    created: datetime
    models: Tuple[str]
    videos: Tuple[str]


def load_trained() -> Iterable[Game]:
    data_path: str = get_setting("data_path")  # type: ignore

    games: List[Game] = []
    for dir in os.listdir(data_path):
        d = path.join(data_path, dir)

        # Ignore anything that is not a folder
        if not path.isdir(d):
            continue

        # If the metadata does not exist also ignore it
        metadata_path = path.join(d, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        metadata: Game = json.loads(open(metadata_path).read())

        videos = tuple(path.abspath(p) for p in os.listdir(path.join(d, "videos")))
        models = tuple(path.abspath(p) for p in os.listdir(path.join(d, "models")))

        metadata["videos"] = videos  # type: ignore
        metadata["models"] = models  # type: ignore

        games.append(metadata)

    return games

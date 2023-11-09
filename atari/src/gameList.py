from json import load
from typing import TypedDict, List


class GameDict(TypedDict):
    name: str
    env: str
    slug: str
    description: str
    default_mode: str
    default_difficulty: str
    url: str
    modes: List[str]
    difficulties: List[str]


gameList: dict[str, GameDict]
with open("src/assets/games.json", "r") as f:
    gameList = load(f)

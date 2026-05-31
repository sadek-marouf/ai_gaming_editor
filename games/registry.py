# games/registry.py

from games.base_profile import BaseGameProfile
from games.pubg import PUBGProfile
from core.logger import get_logger

logger = get_logger("GAME_REGISTRY")

GAME_REGISTRY = {
    "generic": BaseGameProfile,
    "pubg": PUBGProfile,
}


def get_game_profile(name):
    name = name.lower().strip()
    cls = GAME_REGISTRY.get(name)

    if cls is None:
        logger.warning(
            f"Game '{name}' not found. "
            f"Available: {list(GAME_REGISTRY.keys())}. "
            f"Using generic profile."
        )
        cls = BaseGameProfile

    profile = cls()
    logger.info(f"Loaded game profile: {profile.DISPLAY_NAME}")
    return profile


def list_games():
    return {
        name: cls.DISPLAY_NAME
        for name, cls in GAME_REGISTRY.items()
    }

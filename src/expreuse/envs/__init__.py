from .base import Application


def load_application(name: str, **options) -> Application:
    """Open an application by name ('alfworld' or 'scienceworld'); the simulator package must be installed."""
    if name == "alfworld":
        from .alfworld import AlfworldApplication
        return AlfworldApplication(**options)
    if name == "scienceworld":
        from .scienceworld import ScienceWorldApplication
        return ScienceWorldApplication(**options)
    raise ValueError(f"unknown application: {name}")

"""Robot configurations, see README.md."""

import importlib


def load(name: str):
    return importlib.import_module(f"{__name__}.{name}")

"""Sensor models, see README.md."""

import importlib

from .sensor_model import REQUIRED_METADATA, SensorModel


def load(name: str) -> type[SensorModel]:
    """The SensorModel subclass defined in sensors/<name>.py."""
    module = importlib.import_module(f"{__name__}.{name}")
    models = [
        value for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, SensorModel) and value.__module__ == module.__name__
    ]
    if len(models) != 1:
        raise ValueError(f"sensors/{name}.py must define one SensorModel subclass, it defines {len(models)}.")
    return models[0]

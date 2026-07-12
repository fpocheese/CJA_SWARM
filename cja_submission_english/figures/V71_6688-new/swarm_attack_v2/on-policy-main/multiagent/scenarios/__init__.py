from importlib import import_module
from types import ModuleType


def load(scenario_name: str) -> ModuleType:
    """Load a scenario module from local MPE scenarios.

    Parameters
    ----------
    scenario_name: str
        Scenario filename, e.g. ``simple_world_comm_3d.py`` or ``simple_world_comm_3d``.
    """
    if scenario_name.endswith('.py'):
        scenario_name = scenario_name[:-3]
    return import_module(f"onpolicy.envs.mpe.scenarios.{scenario_name}")


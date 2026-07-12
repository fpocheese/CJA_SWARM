"""Compatibility shim for local multiagent-particle-like package.

This project reuses modified files under ``onpolicy.envs.mpe``. During
evaluation we import modules through ``multiagent.*``, so provide wrappers that
delegate to the local implementation to avoid external dependency/runtime
path issues.
"""

from . import scenarios  # noqa: F401


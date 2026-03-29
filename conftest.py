"""
Pytest configuration: bootstrap the turboquant package for test discovery.

Handles the package import so test files can use:
    from turboquant import TurboQuantMSE, ...
    from turboquant.rotations import WHTRotation, ...

without needing the importlib.util shim in every file.
"""

import importlib.util
import os
import sys

# Prevent pytest from collecting __init__.py as a test module
collect_ignore = ["__init__.py"]
collect_ignore_glob = ["__init__*"]

# Bootstrap the package if not already importable
if "turboquant" not in sys.modules:
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "turboquant",
        os.path.join(_pkg_dir, "__init__.py"),
        submodule_search_locations=[_pkg_dir],
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["turboquant"] = _mod
    _spec.loader.exec_module(_mod)


def pytest_ignore_collect(collection_path, config):
    """Prevent pytest from treating __init__.py as importable test module."""
    if collection_path.name == "__init__.py":
        return True
    return False


def pytest_collect_file(parent, file_path):
    """Prevent pytest from collecting __init__.py as a test module."""
    if file_path.name == "__init__.py":
        return None

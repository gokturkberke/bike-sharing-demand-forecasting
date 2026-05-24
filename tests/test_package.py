"""Smoke test: the bike_sharing package is importable and exposes a version."""

import bike_sharing


def test_package_imports():
    assert hasattr(bike_sharing, "__version__")
    assert isinstance(bike_sharing.__version__, str)
    assert bike_sharing.__version__ == "0.1.0"

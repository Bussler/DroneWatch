from __future__ import annotations

import dronewatch
from dronewatch.sim import rust_version


def test_python_package_imports() -> None:
    assert dronewatch.__version__ == "0.1.0"


def test_rust_extension_is_callable_from_python() -> None:
    assert rust_version() == "0.1.0"

import warnings

import pytest

from pyreact.core import HookContext, runtime


@pytest.fixture(autouse=True)
def _isolate_runtime():
    """Runtime/services are process-global; reset them around every test."""
    warnings.simplefilter("ignore", RuntimeWarning)
    runtime.bind_loop(None)
    runtime.set_error_handler(None)
    HookContext._services.clear()
    yield
    runtime.bind_loop(None)
    runtime.set_error_handler(None)
    HookContext._services.clear()

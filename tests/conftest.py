import warnings

import pytest

from pyreact.core import HookContext, runtime


@pytest.fixture(autouse=True)
def _isolate_runtime():
    """The render runtime is process-global; reset it around every test.

    AppRunner sessions own their services; only headless trees (roots created
    without a services dict) use the global ``HookContext._services``.
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    runtime.bind_loop(None)
    runtime.set_error_handler(None)
    HookContext._services.clear()
    yield
    runtime.bind_loop(None)
    runtime.set_error_handler(None)
    HookContext._services.clear()

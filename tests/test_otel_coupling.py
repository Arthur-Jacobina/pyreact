"""pyreact_otel is a library at the border: pyreact never imports it, and the
core imports without the OpenTelemetry packages."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyreact_never_imports_pyreact_otel():
    hits = [str(p) for p in (ROOT / "pyreact").rglob("*.py")
            if "pyreact_otel" in p.read_text(encoding="utf-8")]
    assert hits == []
    code = (
        "import sys, pyreact, pyreact.core, pyreact.boot, pyreact.core.harness\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('pyreact_otel', 'opentelemetry')]\n"
        "assert not bad, bad\n"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


def test_clear_install_hint_without_opentelemetry():
    # Simulate a missing extra: block every opentelemetry import.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'opentelemetry':\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import pyreact.core.harness  # the core is fine\n"
        "try:\n"
        "    import pyreact_otel\n"
        "except ImportError as e:\n"
        "    assert 'pyreact[otel]' in str(e), e\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

"""Every notebook is generated from its source in src/ (scripts/py2nb.py); the two must not drift apart.

Run: pytest tests/test_sources.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted(p for p in (ROOT / "src").glob("*.py") if p.stem[0].isdigit() or p.stem[:2] in ("A_", "B_"))


@pytest.mark.parametrize("src", SOURCES, ids=[p.stem for p in SOURCES])
def test_notebook_matches_source(src, tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "py2nb.py"), str(src), str(out)], check=True)
    cells = lambda p: [c["source"] for c in json.loads(p.read_text())["cells"]]
    assert cells(out) == cells(ROOT / "notebooks" / f"{src.stem}.ipynb"), f"regenerate notebooks/{src.stem}.ipynb"

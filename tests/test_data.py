"""The data manifest is complete and self-consistent (no network needed).

Run: pytest tests/test_data.py
"""
import json
from pathlib import Path

MANIFEST = json.loads((Path(__file__).resolve().parent.parent / "data" / "MANIFEST.json").read_text())


def test_manifest_entries_complete():
    for name, m in MANIFEST.items():
        assert len(m["sha256"]) == 64 and m["rows"] > 0 and m["bytes"] > 0, name
        assert m["queries"], name


def test_files_used_by_notebooks_are_in_manifest():
    """Names passed as string literals, load('name'), must be in the manifest."""
    nb = Path(__file__).resolve().parent.parent / "notebooks"
    used = set()
    for p in nb.glob("*.ipynb"):
        for cell in json.loads(p.read_text())["cells"]:
            src = "".join(cell["source"])
            used |= {s.split("'")[1] for s in src.split("load(")[1:] if s.startswith("'")}
    assert used and used <= set(MANIFEST), used - set(MANIFEST)

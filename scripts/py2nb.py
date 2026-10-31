"""Convert percent-format sources (# %% / # %% [markdown]) to notebooks without outputs.
Usage: python py2nb.py src.py out.ipynb"""
import sys
import nbformat

src, out = sys.argv[1], sys.argv[2]
cells, kind, buf = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(buf).strip("\n")
    if kind == "md":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(text))


import os, re
here = os.path.dirname(os.path.abspath(src))
text = re.sub(r"^#@@(\w+)$", lambda m: open(os.path.join(here, m.group(1).lower() + ".py")).read().rstrip("\n"),
              open(src).read(), flags=re.M)        # inline shared blocks: a line '#@@LOADER' -> contents of loader.py
for line in text.splitlines():
    if line.startswith("# %%"):
        flush()
        kind, buf = ("md" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbformat.write(nb, out)

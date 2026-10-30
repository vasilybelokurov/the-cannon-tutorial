"""Publish the tutorial's data files on Zenodo (instructor only).

Usage:
  python scripts/zenodo_upload.py create    # new draft with a reserved DOI; record id and DOI -> data/zenodo.json
  python scripts/zenodo_upload.py upload    # upload data_build/*, data/MANIFEST.json, data/README.md; set metadata;
                                            # check that Zenodo's MD5 of every file matches the local file
  python scripts/zenodo_upload.py publish   # make the draft public. Irreversible: files can no longer change.

Needs a Zenodo personal access token (scopes deposit:write, deposit:actions) in ~/.config/zenodo/token.
API documentation: https://developers.zenodo.org
"""
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "zenodo.json"
API = "https://zenodo.org/api/deposit/depositions"
TOKEN = (Path.home() / ".config" / "zenodo" / "token").read_text().strip()
AUTH = {"Authorization": f"Bearer {TOKEN}"}

METADATA = {
    "upload_type": "dataset",
    "title": "Stellar labels from spectra with The Cannon: tutorial data (APOGEE DR17 and Gaia DR3 extracts)",
    "creators": [{"name": "Belokurov, Vasily", "affiliation": "Institute of Astronomy, University of Cambridge",
                  "orcid": "0000-0002-0038-9584"}],
    "description": (
        "<p>Data files for the tutorial <a href='https://github.com/vasilybelokurov/the-cannon-tutorial'>Stellar "
        "labels from spectra with The Cannon</a>, which builds the data-driven spectral model of Ness et al. (2015) "
        "step by step: APOGEE DR17 apStar spectra (APO 2.5 m) of samples of red giants, dwarfs and star-cluster "
        "members, with their DR17 labels; Gaia DR3 RVS mean spectra of APOGEE giants, with APOGEE and Gaia GSP-Spec "
        "labels; the strongest lines of the APOGEE DR17 line list; and the continuum pixels of Chapter 3.</p>"
        "<p>Every file is made by scripts/build_data.py in the GitHub repository; MANIFEST.json gives the query, row "
        "count and SHA-256 checksum of each file, and README.md the sample definitions, columns, sources and "
        "acknowledgements. The files are extracts of public SDSS-IV and Gaia data; please also cite the original "
        "data releases listed in README.md.</p>"),
    "access_right": "open",
    "license": "cc-by-4.0",
    "keywords": ["The Cannon", "APOGEE", "Gaia RVS", "stellar spectroscopy", "stellar labels", "tutorial"],
    "version": "v0.1",
    "related_identifiers": [
        {"identifier": "https://github.com/vasilybelokurov/the-cannon-tutorial", "relation": "isSupplementTo",
         "resource_type": "software"},
        {"identifier": "10.1088/0004-637X/808/1/16", "relation": "references", "resource_type": "publication-article"},
        {"identifier": "10.3847/1538-4365/ac4414", "relation": "isDerivedFrom", "resource_type": "publication-article"},
        {"identifier": "10.1051/0004-6361/202243940", "relation": "isDerivedFrom", "resource_type": "publication-article"},
    ],
}


def check(r):
    if not r.ok:
        raise SystemExit(f"Zenodo API error {r.status_code}: {r.text[:500]}")
    return r.json() if r.content else {}


def create():
    if STATE.exists():
        raise SystemExit(f"{STATE} exists: a draft was already created ({json.loads(STATE.read_text())})")
    dep = check(requests.post(API, json={}, headers=AUTH))
    state = {"id": dep["id"], "doi": dep["metadata"]["prereserve_doi"]["doi"],
             "url": f"https://zenodo.org/records/{dep['id']}"}
    STATE.write_text(json.dumps(state, indent=1) + "\n")
    print(state)


def upload():
    state = json.loads(STATE.read_text())
    dep = check(requests.get(f"{API}/{state['id']}", headers=AUTH))
    files = sorted((ROOT / "data_build").iterdir()) + [ROOT / "data" / "MANIFEST.json", ROOT / "data" / "README.md"]
    for f in files:
        print(f"uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        with open(f, "rb") as fh:
            check(requests.put(f"{dep['links']['bucket']}/{f.name}", data=fh, headers=AUTH))
    check(requests.put(f"{API}/{state['id']}", json={"metadata": METADATA}, headers=AUTH))
    remote = {x["filename"]: x["checksum"] for x in check(requests.get(f"{API}/{state['id']}/files", headers=AUTH))}
    for f in files:
        ok = remote.get(f.name) == hashlib.md5(f.read_bytes()).hexdigest()
        print(f"{f.name:28s} MD5 {'ok' if ok else 'MISMATCH'}")
        assert ok, f.name
    print(f"draft ready for review: https://zenodo.org/uploads/{state['id']}  (DOI {state['doi']} reserved)")


def publish():
    state = json.loads(STATE.read_text())
    dep = check(requests.post(f"{API}/{state['id']}/actions/publish", headers=AUTH))
    print(f"published: {dep['links']['record_html']}  DOI {dep['doi']}")


if __name__ == "__main__":
    {"create": create, "upload": upload, "publish": publish}[sys.argv[1]]()

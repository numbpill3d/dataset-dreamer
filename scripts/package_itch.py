"""build / packaging script for itch.io release."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist" / "itch"


def build_itch():
    DIST.mkdir(parents=True, exist_ok=True)

    # build web assets
    assets = ROOT / "assets"
    build_dir = DIST / "dataset-dreamer"
    build_dir.mkdir(parents=True, exist_ok=True)

    # copy all web assets
    shutil.copytree(assets / "static", build_dir / "static")
    shutil.copytree(assets / "templates", build_dir / "templates")

    # copy core python package
    shutil.copytree(ROOT / "src" / "dataset_dreamer", build_dir / "dataset_dreamer")
    shutil.copy2(ROOT / "pyproject.toml", build_dir / "pyproject.toml")

    # create itch manifest
    manifest = {
        "name": "dataset-dreamer",
        "version": "0.1.0",
        "author": "dataset-dreamer",
        "description": "generative data augmentation conditioned on cosmic signals. tabular/timeseries synthesis. 90s aero brutalist UI.",
        "tags": ["generative", "data", "synthesis", "machine-learning", "aesthetic"],
        "platform": "linux",
        "entry": "python -m dataset_dreamer.ui.server --port 8080",
        "screenshots": [],
    }
    (build_dir / "itch-manifest.json").write_text(json.dumps(manifest, indent=2))

    # zip for itch
    zip_path = DIST / "dataset-dreamer-v0.1.0.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for file_path in build_dir.rglob("*"):
            if file_path.is_file():
                arc = file_path.relative_to(DIST)
                z.write(file_path, arc)

    print(f"built itch package: {zip_path}")
    return zip_path


if __name__ == "__main__":
    build_itch()

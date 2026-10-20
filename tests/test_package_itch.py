from __future__ import annotations

import zipfile

from scripts import package_itch


def test_archive_preserves_each_file_path_once(tmp_path, monkeypatch):
    monkeypatch.setattr(package_itch, 'DIST', tmp_path / 'itch')

    archive = package_itch.build_itch()

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()

    assert 'dataset-dreamer/pyproject.toml' in names
    assert 'dataset-dreamer/static/app.js' in names
    assert (
        'dataset-dreamer/static/app.js/static/app.js' not in names
    )

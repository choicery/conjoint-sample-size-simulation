"""Reads the files of one folder of ../../data, stored as a folder or as one ZIP file with the same name."""
import json
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def files(folder, prefix=""):
    """(name without .json, content) of every file whose name starts with prefix, in the order of the names."""
    path = DATA / folder
    if path.is_dir():
        for f in sorted(path.glob(prefix + "*.json"), key=lambda p: p.name):
            yield f.stem, json.loads(f.read_text(encoding="utf-8"))
        return
    with zipfile.ZipFile(DATA / (folder + ".zip")) as archive:
        names = sorted((n.rsplit("/", 1)[-1], n) for n in archive.namelist() if n.endswith(".json") and n.rsplit("/", 1)[-1].startswith(prefix))
        for short, full in names:
            yield short[:-5], json.loads(archive.read(full).decode("utf-8"))


def studies(folder, prefix=""):
    for _, content in files(folder, prefix):
        yield content

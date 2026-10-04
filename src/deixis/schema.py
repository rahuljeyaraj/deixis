"""JSON Schemas for the wire types, for other languages (the web demo's JavaScript, ...).

The schemas are committed under `schema/` (`where-1.json`, ...). A test regenerates them and
fails if they differ, so a schema change is always on purpose: bump the minor version,
regenerate with `python -m deixis schema`, and add a line to CHANGELOG.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deixis.where import WIRE_VERSION, Answer, Ask, Size, Where

SCHEMA_URL = "https://github.com/rahuljeyaraj/deixis/blob/main/schema/"


def _models() -> dict[str, type[BaseModel]]:
    from deixis.profile import Profile
    from deixis.scene import Thing

    models: dict[str, type[BaseModel]] = {
        "where": Where,
        "size": Size,
        "ask": Ask,
        "answer": Answer,
        "thing": Thing,
        "profile": Profile,
    }
    return models


def file_name(name: str) -> str:
    return f"{name}-{WIRE_VERSION.split('.')[0]}.json"


def schemas() -> dict[str, dict[str, Any]]:
    """JSON Schema (draft 2020-12) for each wire type, by name."""
    out: dict[str, dict[str, Any]] = {}
    for name, model in _models().items():
        body = model.model_json_schema()
        out[name] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": SCHEMA_URL + file_name(name),
            **body,
        }
    return out


def schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def write_schemas(directory: str | Path) -> list[Path]:
    """Write every schema to `directory` and return the paths."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in schemas().items():
        path = d / file_name(name)
        path.write_text(schema_text(schema), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    target = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "schema"
    for p in write_schemas(target):
        print(p)
    return 0

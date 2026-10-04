"""Command line: `python -m deixis schema [dir]` writes the JSON Schemas (default: schema/)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["schema"]:
        from deixis.schema import main as schema_main

        return schema_main(args[1:])
    print("usage: python -m deixis schema [dir]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

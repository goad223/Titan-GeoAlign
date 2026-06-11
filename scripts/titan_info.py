#!/usr/bin/env python3
"""titan_info: print a hardware report and the available-formats table.

Usage::

    python scripts/titan_info.py [--json]

Outputs a bilingual (English/Arabic) hardware capability report followed by
a table of every image format the I/O engine knows about and whether it is
usable in the current environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from changemaster.core.hardware import detect_hardware, hardware_report
from changemaster.io_engine.base_reader import ReaderRegistry


def format_table_text(rows: list[dict[str, object]]) -> str:
    """Render the formats table as aligned plain text.

    Args:
        rows: Rows from :meth:`ReaderRegistry.format_table`.

    Returns:
        A multi-line table string with a bilingual header.
    """
    header = ["Format", "Extensions", "Available", "Requires"]
    table = [header] + [
        [
            str(r["format"]),
            str(r["extensions"]),
            "yes | نعم" if r["available"] else "no | لا",
            str(r["requires"]),
        ]
        for r in rows
    ]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = ["=== Supported Formats | الصيغ المدعومة ==="]
    for i, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="titan_info",
        description="Hardware report and supported-formats table | تقرير العتاد وجدول الصيغ",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    profile = detect_hardware()
    rows = ReaderRegistry.format_table()
    if args.json:
        payload = {"hardware": profile.to_dict(), "formats": rows}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(hardware_report(profile))
        print()
        print(format_table_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

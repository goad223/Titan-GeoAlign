#!/usr/bin/env python3
"""titan_inspect: print unified metadata for any supported image or product.

Usage::

    python scripts/titan_inspect.py <path> [--json]

Opens the file/product with the best available reader, prints a bilingual
metadata summary including detected sensor profile, and exits non-zero with
a bilingual error message when the file cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from changemaster.core.exceptions import ChangeMasterError
from changemaster.io_engine.base_reader import open_image
from changemaster.sensors.profiles import SensorRegistry


def inspect(path: str | Path) -> dict[str, object]:
    """Open ``path`` and return its metadata plus detected sensor profile.

    Args:
        path: Image file or product directory.

    Returns:
        Dictionary with ``metadata`` (from ``ImageMetadata.to_dict``) and
        ``sensor_profile`` (profile key and display name).

    Raises:
        ChangeMasterError: For unsupported/missing files or read errors.
    """
    reader = open_image(path)
    try:
        meta = reader.metadata
        profile = SensorRegistry.detect_from_metadata(meta)
        return {
            "metadata": meta.to_dict(),
            "sensor_profile": {
                "name": profile.name,
                "display_name": profile.display_name,
                "platform": profile.platform,
            },
            "summary": meta.summary(),
        }
    finally:
        reader.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on read failure.
    """
    parser = argparse.ArgumentParser(
        prog="titan_inspect",
        description="Inspect a satellite image or product | فحص صورة أو منتج فضائي",
    )
    parser.add_argument("path", help="image file or product directory")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        result = inspect(args.path)
    except ChangeMasterError as exc:
        print(f"Error | خطأ: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload = {k: v for k, v in result.items() if k != "summary"}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(result["summary"])
        profile = result["sensor_profile"]
        assert isinstance(profile, dict)
        print(f"Detected profile | البروفايل المكتشف: {profile['display_name']} ({profile['name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

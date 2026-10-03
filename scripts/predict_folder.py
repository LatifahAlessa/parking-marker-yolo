from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.grouping import IMAGE_EXTENSIONS
from app.pipeline import ParkingCompliancePipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "results.json")
    args = p.parse_args()

    images = sorted(x for x in args.folder.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS)
    payload = [(x.name, x.read_bytes()) for x in images]
    result = ParkingCompliancePipeline().analyze_files(payload)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

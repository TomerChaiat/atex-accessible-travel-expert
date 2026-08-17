"""Upload data/seed/*.json into the Supabase `places` table.

Run sql/schema.sql first, then:

    python scripts/push_supabase.py            # needs SUPABASE_URL + SUPABASE_SERVICE_KEY
    python scripts/push_supabase.py --dry-run

Until this is run the app reads the same JSON files directly, so the local and
deployed catalogues stay identical either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.config import SEED_DIR, settings  # noqa: E402
from atex.httpjson import HttpError, post_json  # noqa: E402
from atex.repository import Place  # noqa: E402

BATCH = 50


def load_places() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(SEED_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("places", []):
            rows.append(Place.from_dict(raw).to_dict())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = settings(refresh=True)
    rows = load_places()
    print(f"Loaded {len(rows)} places from data/seed/")

    synthetic = sum(1 for r in rows if "synthetic" in (r.get("accessibility_source") or ""))
    if synthetic:
        print(f"  note: {synthetic} entries are synthetic demo records, not real venues.")

    if args.dry_run:
        print("Dry run: nothing was sent.")
        return 0

    if cfg.repository_backend != "supabase":
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set. Nothing to push.")
        return 1

    url = f"{cfg.supabase_url}/rest/v1/places?on_conflict=id"
    headers = {
        "apikey": cfg.supabase_service_key,
        "Authorization": f"Bearer {cfg.supabase_service_key}",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    sent = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        try:
            post_json(url, batch, headers=headers, timeout=30.0)
        except HttpError as exc:
            print(f"  ! batch at {start} failed: {exc}")
            return 1
        sent += len(batch)
        print(f"  upserted {sent}/{len(rows)}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

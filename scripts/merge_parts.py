#!/usr/bin/env python
"""Assemble a campaign that was extracted as several ``--part`` jobs.

    python scripts/merge_parts.py            # <out> from extraction.yaml

Each part wrote ``<out>/parts/K/`` with its own shards, manifest, hosts table
and summary.  This puts them together at ``<out>/`` so that everything
downstream -- ``prepare_config.py``, ``diagnose.py``, ``train.py`` -- sees what
it would have seen from a single job, and needs no knowledge that the campaign
was split at all.

**The shards are hard-linked, not copied.**  A campaign is tens of gigabytes and
the parts are already on the filesystem the merged set will live on; a link
costs an inode. They are renamed ``part<K>-<original>`` on the way, which is
what keeps two parts' ``patches-00000.h5`` apart.

**What is summed and what is not.**  The counts and the rejection tallies are
additive and are added. Wall-clock stages are summed too, which gives the
campaign's total compute rather than its elapsed time -- the jobs ran at once.
The correlation length and the visits-per-cell distribution are *not* merged:
they are percentiles and an accumulator, and averaging them would produce a
number that looks authoritative and means nothing. Each part's own summary is
kept under ``parts`` so they are still there to read, and the correlation length
that matters is the one ``prepare_config.py`` measures on the pooled patches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rubin_host_prior.selection import ExtractionConfig

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "extraction.yaml"

#: Summed across parts.  Anything else in a part's summary is either per-part
#: (a distribution, a correlation length) or identical in all of them.
_ADDITIVE = ("counts", "rejection_counts", "seconds")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help=f"run description, for its `out` (default: {DEFAULT_CONFIG})")
    p.add_argument("--out", default=None,
                   help="campaign directory; overrides the config's `out`")
    p.add_argument("--copy", action="store_true",
                   help="copy the shards instead of hard-linking them, for a "
                        "parts directory on a different filesystem")
    return p


def merge(out: Path, copy: bool = False) -> dict:
    parts = sorted((out / "parts").glob("*/summary.json"))
    if not parts:
        raise SystemExit(
            f"no parts under {out / 'parts'}. Run the extraction with "
            f"--part K --of N first; a single-job run needs no merging."
        )
    import pandas as pd

    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    merged: dict = {"merged_from": [], "parts": [], "shards": []}
    manifests, hosts = [], []

    for summary_path in parts:
        part_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        merged["merged_from"].append(str(part_dir))
        merged["parts"].append(summary)
        for key in _ADDITIVE:
            for name, value in (summary.get(key) or {}).items():
                if isinstance(value, (int, float)):
                    merged.setdefault(key, {})
                    merged[key][name] = merged[key].get(name, 0) + value
        for src in sorted((part_dir / "shards").glob("*.h5")):
            dest = shard_dir / f"{part_dir.name}-{src.name}"
            if dest.exists():
                dest.unlink()
            if copy:
                dest.write_bytes(src.read_bytes())
            else:
                dest.hardlink_to(src)
            merged["shards"].append(str(dest))
        for name, into in (("manifest", manifests), ("hosts", hosts)):
            path = part_dir / f"{name}.parquet"
            if path.exists():
                into.append(pd.read_parquet(path))

    # `part` is a per-job label and means nothing for the whole; `stamps_
    # requested` summed back up is the campaign target it was divided from.
    merged.get("counts", {}).pop("part", None)
    for frames, name in ((manifests, "manifest"), (hosts, "hosts")):
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(
                out / f"{name}.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(merged, indent=2, default=str))
    return merged


def main() -> None:
    args = parser().parse_args()
    out = Path(args.out or ExtractionConfig.load(args.config).out)
    merged = merge(out, copy=args.copy)
    counts = merged.get("counts", {})
    print(f"merged {len(merged['merged_from'])} parts into {out}")
    print(f"  {len(merged['shards'])} shards, "
          f"{counts.get('stamps_accepted', 0):,} accepted stamps of "
          f"{counts.get('stamps_attempted', 0):,} attempted")
    print(f"  wrote {out}/manifest.parquet, {out}/hosts.parquet, {out}/summary.json")
    print(f"\n{out}/shards is now what --shards should point at.")


if __name__ == "__main__":
    main()

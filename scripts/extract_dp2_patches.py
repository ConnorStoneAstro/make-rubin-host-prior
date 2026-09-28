#!/usr/bin/env python
"""Extract a ``deep_coadd`` patch training set from DP2.

Runs on NERSC, inside the LSST stack.

    python scripts/extract_dp2_patches.py --config extraction.yaml

Everything the run does comes from that file -- where to look, what to keep, how
much of it to write.  This script has no defaults of its own, so a run is
reproducible from a thing you can read, diff and check in, and there is no
possibility of a command-line flag and a config key disagreeing.

Writes ``<out>/shards/*.h5``, ``<out>/manifest.parquet``, ``<out>/hosts.parquet``
and ``<out>/summary.json``.  Read ``rejection_counts`` in the summary before
trusting the set: what the gate throws away *is* the selection function.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rubin_host_prior.selection import ExtractionConfig
from rubin_host_prior.rubin.extract import (
    describe_summary,
    extract_patches,
    open_butler,
)

#: Shipped with the repository, not generated.
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "extraction.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help=f"run description (default: {DEFAULT_CONFIG})")
    p.add_argument("--repo", default="dp2", help="butler repo alias or path")
    p.add_argument("--collection", default="dp2")
    p.add_argument("--part", type=int, default=None, metavar="K",
                   help="run only part K of a campaign split across --of jobs. "
                        "Whole patches go to a part, so each patch's butler "
                        "reads happen once across the whole campaign; results "
                        "land in <out>/parts/K and scripts/merge_parts.py "
                        "assembles them")
    p.add_argument("--of", type=int, default=None, metavar="N",
                   help="how many parts the campaign is split into. n_stamps in "
                        "the config is the campaign total; a part takes its "
                        "share, as train.steps is the length of a whole "
                        "chunked run rather than one chunk")
    p.add_argument("--no-plots", action="store_true",
                   help="skip the diagnostic figures written to <out>/diagnostics")
    p.add_argument("--verbose", "-v", action="count", default=0,
                   help="DEBUG instead of the default INFO")
    args = p.parse_args()

    # INFO by default: an extraction runs for tens of minutes and a run that
    # says nothing is a run you cannot tell from a hung one.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Which slice of a campaign this job runs is a property of the job, not of
    # the run -- like --devices on train.py -- so it is a flag and not a config
    # key.  Putting it in extraction.yaml would mean N files differing in one
    # number, which is exactly the thing that file exists to avoid.
    if (args.part is None) != (args.of is None):
        p.error("--part and --of go together")
    part = None if args.part is None else (args.part, args.of)
    if part is not None and not 0 <= args.part < args.of:
        p.error(f"--part must be in [0, {args.of}), got {args.part}")

    config = ExtractionConfig.load(args.config)
    print(f"config: {args.config}")
    print(config.describe())

    out_dir = config.out if part is None else f"{config.out}/parts/{args.part}"
    if part is not None:
        print(f"part {args.part} of {args.of} -> {out_dir}")

    butler = open_butler(args.repo, args.collection)
    summary = extract_patches(
        butler,
        out_dir=out_dir,
        part=part,
        ra=config.sky.ra,
        dec=config.sky.dec,
        radius_deg=config.sky.radius_deg,
        bands=config.stamps.bands,
        native_size=config.stamps.native_size,
        patches_per_shard=config.stamps.patches_per_shard,
        host_cache=config.catalogue.cache,
        limit_hosts=config.catalogue.limit_hosts,
        tap_url=config.catalogue.tap_url,
        n_stamps=config.run.n_stamps,
        seed=config.run.seed,
        prefix=config.run.prefix,
        selection=config,
    )
    print(json.dumps(summary, indent=2, default=str))
    print()
    print(describe_summary(summary))

    if not args.no_plots:
        # Extraction-side figures only: the loader figures need the softening
        # scales, which prepare_config.py has not produced yet.  Never let a
        # plotting failure cost a completed extraction.
        try:
            _write_plots(Path(config.out), config.stamps.bands)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "diagnostic plots failed (%r); the data is fine", exc)


def _write_plots(out: Path, bands) -> None:
    import pandas as pd

    from rubin_host_prior import plots
    from rubin_host_prior.data import ShardSet

    shards = ShardSet.from_dir(out / "shards")
    hosts = pd.read_parquet(out / "hosts.parquet")
    manifest = pd.read_parquet(out / "manifest.parquet")
    band = "r" if "r" in bands else bands[0]
    for path in plots.make_all(shards, hosts=hosts, manifest=manifest,
                               out_dir=out / "diagnostics", band=band):
        print(f"  wrote {path}")
    print("  run scripts/diagnose.py --config <config.json> for the loader figures")


if __name__ == "__main__":
    main()

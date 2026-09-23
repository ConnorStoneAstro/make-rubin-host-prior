#!/usr/bin/env python
"""Write diagnostic figures for an extracted training set.

    python scripts/diagnose.py --shards data/ecdfs/shards --out data/ecdfs/diagnostics

Reads ``hosts.parquet`` and ``manifest.parquet`` from the shards' parent
directory when they are there, so after a normal extraction run the default
paths just work.  Pass ``--config`` to also render what the loader will yield;
without it only the extraction-side figures are produced, since the training
representation is undefined until the softening scales are known.

Figures, in rough order of how often they catch something:

  rejections.png     is the gate discarding the bright dense hosts the project
                     exists to model?
  transform.png      does the log representation look the way the arithmetic
                     says -- sky at log(log 2), sources clear of it?
  training_batch.png what the network actually receives, augmentation and all
  cutouts.png        raw stamps in units of their own sky noise
  hosts.png          the selected population: size, magnitude, ellipticity,
                     band, sky noise, depth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rubin_host_prior import plots
from rubin_host_prior.config import Config
from rubin_host_prior.data import (
    LogFluxTransform,
    PatchDataset,
    ShardSet,
    estimate_band_softening,
    pool_shards,
)


def _find(explicit: str | None, stem: str, *dirs: Path) -> Path | None:
    """Locate a sidecar table. Extraction writes these beside the shard
    directory; the synthetic generator writes them inside it, so look in both."""
    if explicit:
        return Path(explicit)
    for d in dirs:
        for suffix in (".parquet", ".csv"):
            candidate = d / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _read_table(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        import pandas as pd

        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        print(f"  could not read {path.name}: {exc}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shards", required=True, help="directory of *.h5 shards")
    p.add_argument("--out", default=None, help="default: <shards>/../diagnostics")
    p.add_argument(
        "--config", default=None, help="config JSON; without it the loader figures are skipped"
    )
    p.add_argument("--hosts", default=None, help="default: <shards>/../hosts.parquet")
    p.add_argument("--manifest", default=None, help="default: <shards>/../manifest.parquet")
    p.add_argument("--band", default="r", help="band for the magnitude panel")
    p.add_argument("--n-cutouts", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    shard_dir = Path(args.shards)
    root = shard_dir.parent
    out = Path(args.out) if args.out else root / "diagnostics"

    shards = ShardSet.from_dir(shard_dir)
    print(f"{len(shards)} patches, {shards.native_size}px native, bands {shards.bands}")

    hosts = _read_table(_find(args.hosts, "hosts", root, shard_dir))
    manifest = _read_table(_find(args.manifest, "manifest", root, shard_dir))
    print(
        f"  hosts table: {'found' if hosts is not None else 'not found'}"
        f"   manifest: {'found' if manifest is not None else 'not found'}"
    )

    dataset = None
    if args.config:
        config = Config.load(args.config)
        if not config.transform.band_softening:
            # A config written before prepare_config.py ran: derive the scales
            # here so the loader figures are still available.
            pooled, pooled_bands = pool_shards(shards, config, n=512)
            config.transform.band_softening = estimate_band_softening(
                pooled,
                pooled_bands,
                config.transform.softening_sigma,
                bands=shards.bands,
            )
        dataset = PatchDataset.from_shards(
            shards, config, LogFluxTransform.from_config(config.transform)
        )
    else:
        print("no --config: skipping the loader figures")

    written = plots.make_all(
        shards,
        dataset=dataset,
        hosts=hosts,
        manifest=manifest,
        out_dir=out,
        band=args.band,
        n_cutouts=args.n_cutouts,
        seed=args.seed,
    )
    for path in written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()

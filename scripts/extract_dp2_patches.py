#!/usr/bin/env python
"""Extract a ``deep_coadd`` patch training set from DP2.

Runs on NERSC, inside the LSST stack.

    python scripts/extract_dp2_patches.py --out data/ecdfs --bands r i --n-hosts 2000

Writes ``<out>/shards/*.h5``, ``<out>/manifest.parquet``, ``<out>/summary.json``.
Read the rejection counts in the summary before trusting the set: if bright
dense centres are being rejected, the training set is biased against the regime
this project cares about and the gate needs loosening.

Coadds give at most one patch per band per host, so ``--n-hosts`` sets the
training-set size fairly directly: expect roughly ``n_hosts * len(bands)``
patches before rejections.

Early DP2 publishes ``deep_coadd`` and nothing else -- no ``visit_image``, no
``difference_image`` -- which suits this project, since the prior trains on
coadds anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rubin_host_prior.rubin.quality import MAX_VARIANCE_STEP
from rubin_host_prior.rubin.extract import (
    COLLECTION,
    ECDFS,
    REPO,
    extract_patches,
    open_butler,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--repo", default=REPO, help="butler repo alias or path")
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--ra", type=float, default=ECDFS[0])
    p.add_argument("--dec", type=float, default=ECDFS[1])
    p.add_argument("--radius-deg", type=float, default=0.3,
                   help="field radius searched for coadd patches")
    p.add_argument("--bands", nargs="+", default=["u", "g", "r", "i", "z", "y"])
    p.add_argument(
        "--native-size",
        type=int,
        default=416,
        help="native pixels per stamp. This is what caps the training patch "
        "size: out_size <= native_size // pool_factor. 416 targets 128 px "
        "patches (56%% of each clears the loss crop) with slack left for "
        "translation augmentation",
    )
    p.add_argument("--n-hosts", type=int, default=8000)
    p.add_argument(
        "--min-reff-arcsec", type=float, default=1.0,
        help="host cModel half-light major axis floor. The catalogue is mostly "
             "galaxies a pixel or two across, which carry no structure to learn",
    )
    p.add_argument(
        "--max-variance-step", type=float, default=MAX_VARIANCE_STEP,
        help="reject a stamp whose block variance floors differ by more than "
             "this ratio. Cell-based coadds step in depth at cell edges and no "
             "mask plane flags it; set high to keep them and cut later from the "
             "manifest, which records the ratio either way",
    )
    p.add_argument("--jitter-arcsec", type=float, default=4.0)
    p.add_argument("--patches-per-shard", type=int, default=1024)
    p.add_argument("--max-patches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plots", action="store_true",
                   help="skip the diagnostic figures written to <out>/diagnostics")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    butler = open_butler(args.repo, args.collection)
    summary = extract_patches(
        butler,
        out_dir=args.out,
        ra=args.ra,
        dec=args.dec,
        radius_deg=args.radius_deg,
        bands=args.bands,
        native_size=args.native_size,
        n_hosts=args.n_hosts,
        min_reff_arcsec=args.min_reff_arcsec,
        gate_kwargs={"max_variance_step": args.max_variance_step},
        jitter_arcsec=args.jitter_arcsec,
        patches_per_shard=args.patches_per_shard,
        max_patches=args.max_patches,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, default=str))

    if not args.no_plots:
        # Extraction-side figures only: the loader figures need the softening
        # scales, which prepare_config.py has not produced yet.  Never let a
        # plotting failure cost a completed extraction.
        try:
            _write_plots(Path(args.out), args.bands)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "diagnostic plots failed (%s); the extraction itself is fine", exc
            )


def _write_plots(root: Path, bands) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from rubin_host_prior import plots
    from rubin_host_prior.data import ShardSet

    shards = ShardSet.from_dir(root / "shards")
    hosts = manifest = None
    try:
        import pandas as pd

        if (root / "hosts.parquet").exists():
            hosts = pd.read_parquet(root / "hosts.parquet")
        if (root / "manifest.parquet").exists():
            manifest = pd.read_parquet(root / "manifest.parquet")
    except Exception:
        pass
    band = "r" if "r" in bands else bands[0]
    for path in plots.make_all(shards, hosts=hosts, manifest=manifest,
                               out_dir=root / "diagnostics", band=band):
        print(f"  wrote {path}")
    print("  run scripts/diagnose.py --config <config.json> for the loader figures")


if __name__ == "__main__":
    main()

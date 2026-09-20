#!/usr/bin/env python
"""Extract a ``deep_coadd`` patch training set from DP1.

Runs on NERSC, inside the LSST stack.

    python scripts/extract_dp1_patches.py --out data/ecdfs --bands r i --n-hosts 2000

Writes ``<out>/shards/*.h5``, ``<out>/manifest.parquet``, ``<out>/summary.json``.
Read the rejection counts in the summary before trusting the set: if bright
dense centres are being rejected, the training set is biased against the regime
this project cares about and the gate needs loosening.

Coadds give at most one patch per band per host, so ``--n-hosts`` sets the
training-set size fairly directly: expect roughly ``n_hosts * len(bands)``
patches before rejections.
"""

from __future__ import annotations

import argparse
import json
import logging

from rubin_host_prior.rubin.extract import ECDFS, extract_patches, open_butler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--repo", default="dp1", help="butler repo alias or path")
    p.add_argument("--collection", default="LSSTComCam/DP1")
    p.add_argument("--ra", type=float, default=ECDFS[0])
    p.add_argument("--dec", type=float, default=ECDFS[1])
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
    p.add_argument("--psf-size", type=int, default=41)
    p.add_argument("--n-hosts", type=int, default=8000)
    p.add_argument("--jitter-arcsec", type=float, default=4.0)
    p.add_argument("--patches-per-shard", type=int, default=1024)
    p.add_argument("--max-patches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
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
        bands=args.bands,
        native_size=args.native_size,
        psf_size=args.psf_size,
        n_hosts=args.n_hosts,
        jitter_arcsec=args.jitter_arcsec,
        patches_per_shard=args.patches_per_shard,
        max_patches=args.max_patches,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

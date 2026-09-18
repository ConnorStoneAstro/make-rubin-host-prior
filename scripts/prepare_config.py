#!/usr/bin/env python
"""Derive the per-band offsets and a sigma range from extracted shards.

    python scripts/prepare_config.py --shards data/ecdfs_r/shards --out config.json

The offsets and the sigma range are not free hyperparameters -- they follow from
the data's noise level and dynamic range.  This script measures both, writes a
config, and prints the diagnostics you should look at before training:

* ``sky_scatter`` should come out near ``1 / k_sigma``.  If it does not, the
  offsets are wrong and the bands will not be on a common footing.
* ``clipped_fraction`` should be ~1e-5 or smaller.  Larger means either
  ``k_sigma`` is too small or the shards contain over-subtraction artefacts the
  gate missed.
"""

from __future__ import annotations

import argparse
import json

from rubin_host_prior.config import Config
from rubin_host_prior import geometry
from rubin_host_prior.data import (
    LogFluxTransform,
    PatchDataset,
    ShardSet,
    estimate_band_offsets,
    context_advice,
    suggest_sigma_range,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", required=True, help="directory of *.h5 shards")
    p.add_argument("--out", required=True, help="config JSON to write")
    p.add_argument("--base-config", default=None, help="config to start from")
    p.add_argument("--k-sigma", type=float, default=5.0)
    p.add_argument("--pool-factor", type=int, default=3)
    p.add_argument("--out-size", type=int, default=64)
    p.add_argument("--n-stats", type=int, default=512)
    p.add_argument("--pooled-cache", default=None, help="also build a pooled cache")
    args = p.parse_args()

    shards = ShardSet.from_dir(args.shards)
    config = Config.load(args.base_config) if args.base_config else Config()
    config.patch.pool_factor = args.pool_factor
    config.patch.out_size = args.out_size
    config.patch.nominal_crop = args.out_size * args.pool_factor
    config.patch.native_size = max(shards.native_size, config.patch.nominal_crop)
    config.transform.k_sigma = args.k_sigma
    config.transform.band_offsets = estimate_band_offsets(
        shards.load("variance"),
        shards.meta["band_idx"],
        pool_factor=args.pool_factor,
        k_sigma=args.k_sigma,
        bands=shards.bands,
    )

    transform = LogFluxTransform.from_config(config.transform, shards.bands)
    dataset = PatchDataset.from_shards(shards, config, transform)
    stats = dataset.stats(args.n_stats)
    sigma_min, sigma_max = suggest_sigma_range(stats)
    config.sde.sigma_min = round(sigma_min, 5)
    config.sde.sigma_max = round(sigma_max, 3)
    config.save(args.out)

    print(json.dumps({"n_patches": len(shards), "stats": stats,
                      "band_offsets_nJy": config.transform.band_offsets,
                      "sigma_min": config.sde.sigma_min,
                      "sigma_max": config.sde.sigma_max}, indent=2))

    # The correlation length, measured on the pooled log-space patches the model
    # actually sees.  This is the authoritative version -- the one in the
    # extraction summary is native-resolution flux and is contaminated by the PSF.
    lo, hi = config.usable_size_range()
    print(f"\nusable training sizes with {config.energy.n_layers} layers and "
          f"{config.patch.native_size} px stamps: {lo} .. {hi}")
    for w in config.check_sizes():
        print(f"  WARNING: {w}")

    cl = dataset.correlation_length(args.n_stats)
    margin = geometry.loss_margin(config.energy.n_layers, config.energy.kernel_size)
    print(f"\ncorrelation length (pooled, log space, over {cl['n_patches']} patches)")
    print(f"  profile: " + " ".join(
        f"{v:.2f}" for v in cl["profile"][:10]))
    print(f"  {cl['noise_fraction']:.0%} of the variance is the zero-lag noise "
          f"delta (excluded from xi)")
    print(f"  {context_advice(cl['xi'], margin)}")
    if cl["truncated"]:
        print("  WARNING: the patches never decorrelate within their own size, so "
              "xi is a lower bound. Extract larger patches to measure it.")
    expected = 1.0 / args.k_sigma
    if not 0.5 * expected < stats["sky_scatter"] < 2.0 * expected:
        print(
            f"\nWARNING: sky_scatter {stats['sky_scatter']:.3f} is far from the "
            f"expected 1/k_sigma = {expected:.3f}. The band offsets are probably "
            f"wrong -- check the variance planes in the shards."
        )
    if stats["clipped_fraction"] > 1e-4:
        print(
            f"\nWARNING: {stats['clipped_fraction']:.2%} of pixels hit the log "
            f"floor. Raise k_sigma or tighten the artefact gate."
        )
    if args.pooled_cache:
        path = dataset.build_pooled_cache(args.pooled_cache)
        print(f"\npooled cache: {path}")


if __name__ == "__main__":
    main()

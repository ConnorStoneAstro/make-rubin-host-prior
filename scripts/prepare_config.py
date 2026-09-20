#!/usr/bin/env python
"""Derive the per-band offsets and a sigma range from extracted shards.

    python scripts/prepare_config.py --shards data/ecdfs_r/shards --out config.json

The offsets and the sigma range are not free hyperparameters -- they follow from
the data's noise level and dynamic range.  This script measures both, writes a
config, and prints the diagnostics you should look at before training:

* ``sky_scatter`` should match ``expected_sky_scatter(softening_sigma)``.  If it
  does not, the per-band softening scales are wrong and the bands will not be on
  a common footing, which breaks the single band-agnostic prior.
* the sky pedestal should stay below one sigma, and the flux above which the
  exponential model map is accurate should sit below anything you care about
  photometrically.
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
    estimate_band_softening,
    expected_sky_scatter,
    pool_shards,
    context_advice,
    suggest_sigma_range,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", required=True, help="directory of *.h5 shards")
    p.add_argument("--out", required=True, help="config JSON to write")
    p.add_argument("--base-config", default=None, help="config to start from")
    p.add_argument("--softening-sigma", type=float, default=1.0,
                   help="softplus scale in units of pooled sky noise. Sets both "
                        "the sky pedestal (0.693x this, in sigma) and the flux "
                        "above which the exponential model map is accurate")
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
    config.transform.softening_sigma = args.softening_sigma
    # Measured from pooled patches, not derived from the variance plane:
    # coadd pixel noise is correlated, so pooling reduces it by less than
    # pool_factor and the derived value would be badly low.
    pooled, pooled_bands = pool_shards(shards, config, n=args.n_stats)
    config.transform.band_softening = estimate_band_softening(
        pooled,
        pooled_bands,
        softening_sigma=args.softening_sigma,
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
                      "band_softening_nJy": config.transform.band_softening,
                      "sigma_min": config.sde.sigma_min,
                      "sigma_max": config.sde.sigma_max}, indent=2))

    # The correlation length, measured on the pooled log-space patches the model
    # actually sees.  This is the authoritative version -- the one in the
    # extraction summary is native-resolution flux and is contaminated by the PSF.
    t = LogFluxTransform.from_config(config.transform, shards.bands)
    ss = config.transform.softening_sigma
    print(f"\nlog transform:  x = log(softplus(f/s))/c,  model map f = s*exp(c*x)")
    print(f"  softening s = {ss:.2f} x pooled sky noise")
    print(f"  model sky pedestal: {t.sky_pedestal * ss:.2f} sigma"
          f"{'  (below the noise, good)' if t.sky_pedestal * ss < 1 else '  (ABOVE the noise)'}")
    print(f"  exponential map accurate within 1% above "
          f"{t.accurate_above(0.01) * ss:.1f} sigma, 0.1% above "
          f"{t.accurate_above(0.001) * ss:.1f} sigma")
    print(f"  deepest measured pixel: {stats['deepest_flux_sigma']:.1f} sigma "
          f"-> representable (softplus has no floor)")
    predicted = expected_sky_scatter(ss, config.transform.log_scale)
    print(f"  sky scatter in x: {stats['sky_scatter']:.3f} measured vs "
          f"{predicted:.3f} predicted")
    if not 0.5 * predicted < stats["sky_scatter"] < 2.0 * predicted:
        print("  WARNING: measured sky scatter is far from prediction -- the "
              "per-band softening scales are probably wrong, which would put the "
              "bands on different footings.")

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
    if args.pooled_cache:
        path = dataset.build_pooled_cache(args.pooled_cache)
        print(f"\npooled cache: {path}")


if __name__ == "__main__":
    main()

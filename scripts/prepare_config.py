#!/usr/bin/env python
"""Derive the per-band offsets and a sigma range from extracted shards.

    python scripts/prepare_config.py --shards data/ecdfs_r/shards --out config.json

The offsets and the sigma range are not free hyperparameters -- they follow from
the data's noise level and dynamic range.  This script measures both, writes a
config, and prints the diagnostics you should look at before training:

* ``sky_scatter`` should match ``expected_sky_scatter(softening_sigma)``.  If it
  does not, the softening scale is wrong.  It is a check on a *typical* width:
  one scale serves every band, so a band deeper or shallower than the scale was
  measured from scatters proportionally less or more.
* the flux above which the exponential model map is accurate should sit below
  anything you care about photometrically.  The sky pedestal is *meant* to sit
  above one sigma -- suppressing the sky is the point of the softening, and the
  likelihood, not the prior, is what models the noise.
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
    estimate_softening,
    expected_sky_scatter,
    pool_shards,
    context_advice,
    suggest_sigma_range,
)


def _band_counts(shards) -> dict[str, int]:
    """How many patches each band actually contributed."""
    import numpy as np

    idx = np.asarray(shards.meta["band_idx"], dtype=int)
    return {b: int(np.sum(idx == i)) for i, b in enumerate(shards.bands)}


def parser() -> argparse.ArgumentParser:
    """Built separately so a test can ask what a flag defaults to.

    ``--softening-sigma`` must default to None rather than to a number: it
    overrides the config, and a numeric default assigned unconditionally is how
    this script spent a while silently resetting ``softening_sigma`` to 1.0
    whatever the config said.  A test that greps this file for the literal is a
    test of its formatting; a test that asks the parser is a test of the rule.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", required=True, help="directory of *.h5 shards")
    p.add_argument("--out", required=True, help="config JSON to write")
    p.add_argument("--base-config", default=None, help="config to start from")
    p.add_argument(
        "--softening-sigma",
        type=float,
        default=None,
        help="softplus scale in units of pooled sky noise, s = this x "
        "noise. Sets the sky pedestal (0.693x this, in sigma) "
        "and the flux above which the exponential model map is "
        "accurate. Omit to use the config's own value",
    )
    p.add_argument("--pool-factor", type=int, default=3)
    p.add_argument("--out-size", type=int, default=64)
    p.add_argument("--n-stats", type=int, default=512)
    p.add_argument("--pooled-cache", default=None, help="also build a pooled cache")
    return p


def main() -> None:
    args = parser().parse_args()

    shards = ShardSet.from_dir(args.shards)
    config = Config.load(args.base_config) if args.base_config else Config()
    config.patch.pool_factor = args.pool_factor
    config.patch.out_size = args.out_size
    config.patch.nominal_crop = args.out_size * args.pool_factor
    config.patch.native_size = max(shards.native_size, config.patch.nominal_crop)
    # Only when asked.  Assigning the flag's default unconditionally is how
    # this script spent a while quietly resetting softening_sigma to the 1.0 of
    # the old preserve-the-noise formulation, whatever the config said.
    if args.softening_sigma is not None:
        config.transform.softening_sigma = args.softening_sigma
    softening_sigma = config.transform.softening_sigma
    # Measured from pooled patches, not derived from the variance plane:
    # coadd pixel noise is correlated, so pooling reduces it by less than
    # pool_factor and the derived value would be badly low.
    pooled, _ = pool_shards(shards, config, n=args.n_stats)
    config.transform.softening = estimate_softening(pooled, softening_sigma)

    # One scale for every band, so the band counts are reported and nothing
    # more: a band with no patches no longer leaves a hole in the transform.
    # It is still worth seeing, because a band missing entirely usually means
    # the extraction stopped at its target before reaching it.
    counts = _band_counts(shards)
    print("patches per band: " + ", ".join(f"{b}={counts.get(b, 0)}" for b in shards.bands))
    absent = [b for b in shards.bands if counts.get(b, 0) == 0]
    if absent:
        print(
            f"  NOTE: no patches in {absent}. Check the manifest's "
            f"rejection_counts -- a band missing entirely is usually a run that "
            f"stopped at its target before reaching it, or coadds that do not "
            f"exist for those tracts."
        )

    transform = LogFluxTransform.from_config(config.transform)
    dataset = PatchDataset.from_shards(shards, config, transform)
    stats = dataset.stats(args.n_stats)
    sigma_min, sigma_max = suggest_sigma_range(stats)
    config.sde.sigma_min = round(sigma_min, 5)
    config.sde.sigma_max = round(sigma_max, 3)
    # x is absolute log flux, so the data is centred wherever the fluxes put it
    # -- near +3 at DP2 depths, not near zero.  VE keeps the mean, so the t=1
    # marginal is centred here and prior_sample has to start from the same place.
    config.sde.data_mean = round(float(stats["mean"]), 4)
    config.save(args.out)

    print(
        json.dumps(
            {
                "n_patches": len(shards),
                "patches_per_band": counts,
                "stats": stats,
                "softening_nJy": config.transform.softening,
                "sigma_min": config.sde.sigma_min,
                "sigma_max": config.sde.sigma_max,
            },
            indent=2,
        )
    )

    # The correlation length, measured on the pooled log-space patches the model
    # actually sees.  This is the authoritative version -- the one in the
    # extraction summary is native-resolution flux and is contaminated by the PSF.
    t = LogFluxTransform.from_config(config.transform)
    ss = config.transform.softening_sigma
    print(f"\nlog transform:  x = log(s*softplus(f/s)),  model map f = exp(x)")
    print(f"  softening s = {t.softening:.1f} nJy ({ss:.2f} x pooled sky noise)")
    print(f"  bright flux passes through as log(f), the same x in every band")
    print(f"  sky sits at log(s*log2) = {t.sky_level:.2f}, also the same in "
          f"every band")
    print(
        f"  data mean x = {config.sde.data_mean:.3f}, which is where " f"prior_sample starts from"
    )
    print(
        f"  sky pedestal: {t.sky_pedestal * ss:.2f} sigma -- pixels within "
        f"the noise compress towards this, which is the point"
    )
    print(
        f"  exponential map accurate within 1% above "
        f"{t.accurate_above(0.01) * ss:.1f} sigma, 0.1% above "
        f"{t.accurate_above(0.001) * ss:.1f} sigma"
    )
    print(
        f"  deepest measured pixel: {stats['deepest_flux_sigma']:.1f} sigma "
        f"-> representable (softplus has no floor)"
    )
    predicted = expected_sky_scatter(ss)
    print(
        f"  sky scatter in x: {stats['sky_scatter']:.3f} measured vs " f"{predicted:.3f} predicted"
    )
    if not 0.5 * predicted < stats["sky_scatter"] < 2.0 * predicted:
        print(
            "  WARNING: measured sky scatter is far from prediction -- the "
            "softening scale is probably wrong."
        )

    lo, hi = config.usable_size_range()
    print(
        f"\nusable training sizes with {config.energy.n_layers} layers and "
        f"{config.patch.native_size} px stamps: {lo} .. {hi}"
    )
    for w in config.check_sizes():
        print(f"  WARNING: {w}")

    cl = dataset.correlation_length(args.n_stats)
    margin = geometry.loss_margin(config.energy.n_layers, config.energy.kernel_size)
    print(f"\ncorrelation length (pooled, log space, over {cl['n_patches']} patches)")
    print(f"  profile: " + " ".join(f"{v:.2f}" for v in cl["profile"][:10]))
    print(
        f"  {cl['noise_fraction']:.0%} of the variance is the zero-lag noise "
        f"delta (excluded from xi)"
    )
    print(f"  {context_advice(cl['xi'], margin)}")
    if cl["truncated"]:
        print(
            "  WARNING: the patches never decorrelate within their own size, so "
            "xi is a lower bound. Extract larger patches to measure it."
        )
    if args.pooled_cache:
        path = dataset.build_pooled_cache(args.pooled_cache)
        print(f"\npooled cache: {path}")


if __name__ == "__main__":
    main()

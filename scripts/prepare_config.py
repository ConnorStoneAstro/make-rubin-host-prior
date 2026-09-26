#!/usr/bin/env python
"""Derive the per-band offsets and a sigma range from extracted shards.

    python scripts/prepare_config.py --shards data/ecdfs_r/shards --out config.json

The softening scale, the sigma range and the data mean are not free
hyperparameters -- they follow from the data's noise level and dynamic range.
They are ``None`` in ``config.py`` for that reason, and this script fills them
in.  A value already set is kept and reported, never overwritten: setting one
yourself is the only way to override the heuristic, and there is deliberately
no flag for it.

It writes a config and prints the diagnostics you should look at before
training:

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
from pathlib import Path

from rubin_host_prior.config import Config
from rubin_host_prior.selection import ExtractionConfig
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


#: Where the extraction writes its shards, taken from the extraction config
#: rather than written out again here.  ``extract_dp2_patches.py`` writes
#: ``<out>/shards``, and ``out`` has exactly one definition.
DEFAULT_SHARDS = Path(ExtractionConfig.out) / "shards"
DEFAULT_OUT = "config.json"


def _band_counts(shards) -> dict[str, int]:
    """How many patches each band actually contributed."""
    import numpy as np

    idx = np.asarray(shards.meta["band_idx"], dtype=int)
    return {b: int(np.sum(idx == i)) for i, b in enumerate(shards.bands)}


def parser() -> argparse.ArgumentParser:
    """Built separately so a test can ask what a flag defaults to.

    **Every flag that names a config field defaults to None**, and is applied
    only when given.  ``config.py`` holds the defaults for the whole project;
    a number written here as well is a second source of truth, and the two
    drift -- this script spent a while resetting ``softening_sigma`` to the 1.0
    of the old preserve-the-noise design and ``out_size`` to 64, whatever the
    config said, because it assigned its own defaults unconditionally.

    A test that greps this file for the literals is a test of its formatting;
    a test that asks the parser is a test of the rule.

    ``--shards`` and ``--out`` are the exception, because they are not config
    fields at all -- they say where this script reads and writes.  They still
    have defaults, since in practice they are always the same two paths, and
    the shard default follows ``ExtractionConfig.out`` rather than repeating it.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", default=str(DEFAULT_SHARDS),
                   help=f"directory of *.h5 shards (default: {DEFAULT_SHARDS})")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"config JSON to write (default: {DEFAULT_OUT})")
    p.add_argument("--base-config", default=None,
                   help="config to start from. Any measured field it already "
                        "carries (softening, sigma range, data mean) is kept "
                        "rather than re-measured, and the script says which")
    p.add_argument(
        "--softening-sigma",
        type=float,
        default=None,
        help="softplus scale in units of pooled sky noise, s = this x "
        "noise. Sets the sky pedestal (0.693x this, in sigma) "
        "and the flux above which the exponential model map is "
        "accurate. Omit to use the config's own value",
    )
    p.add_argument("--pool-factor", type=int, default=None,
                   help="omit to use the config's own value")
    p.add_argument("--out-size", type=int, default=None,
                   help="pooled training patch size; omit to use the config's "
                        "own value. nominal_crop follows as out_size x "
                        "pool_factor")
    p.add_argument("--n-stats", type=int, default=512,
                   help="patches to measure the statistics from. Not a config "
                        "field: it changes how well this script measures, not "
                        "what the model is")
    return p


def main() -> None:
    args = parser().parse_args()

    shards = ShardSet.from_dir(args.shards)
    config = Config.load(args.base_config) if args.base_config else Config()

    # Only what was actually asked for.  Assigning a flag's own default
    # unconditionally is how this script spent a while resetting out_size to 64
    # and softening_sigma to 1.0 whatever the config said -- the defaults live
    # in config.py, and this script's job is to measure, not to decide.
    if args.pool_factor is not None:
        config.patch.pool_factor = args.pool_factor
    if args.out_size is not None:
        config.patch.out_size = args.out_size
    if args.softening_sigma is not None:
        config.transform.softening_sigma = args.softening_sigma

    # Not a default but an invariant: PatchConfig requires these to agree, and
    # checks it at construction only -- assigning to a field afterwards does not
    # re-run __post_init__.  A no-op unless a flag above moved one of them.
    config.patch.nominal_crop = config.patch.out_size * config.patch.pool_factor
    # Measured, like everything below it: the stamps are as big as the
    # extraction made them.
    config.patch.native_size = shards.native_size
    if shards.native_size < config.patch.nominal_crop:
        raise SystemExit(
            f"the shards are {shards.native_size} px native, but out_size "
            f"{config.patch.out_size} x pool_factor {config.patch.pool_factor} "
            f"needs {config.patch.nominal_crop}. Lower --out-size, or extract "
            f"larger stamps."
        )
    softening_sigma = config.transform.softening_sigma
    # Measured from pooled patches, not derived from the variance plane:
    # coadd pixel noise is correlated, so pooling reduces it by less than
    # pool_factor and the derived value would be badly low.
    pooled, _ = pool_shards(shards, config, n=args.n_stats)
    # Measured only where nothing has been set.  None means "not measured yet";
    # a number means someone chose it, and choosing one is the only way to
    # override the heuristic -- there is deliberately no flag for that.
    kept: list[str] = []
    if config.transform.softening is None:
        config.transform.softening = estimate_softening(pooled, softening_sigma)
    else:
        kept.append(f"transform.softening = {config.transform.softening}")

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
    if config.sde.sigma_min is None:
        config.sde.sigma_min = round(sigma_min, 5)
    else:
        kept.append(f"sde.sigma_min = {config.sde.sigma_min}")
    if config.sde.sigma_max is None:
        config.sde.sigma_max = round(sigma_max, 3)
    else:
        kept.append(f"sde.sigma_max = {config.sde.sigma_max}")
    # x is absolute log flux, so the data is centred wherever the fluxes put it
    # -- near +3 at DP2 depths, not near zero.  VE keeps the mean, so the t=1
    # marginal is centred here and prior_sample has to start from the same place.
    if config.sde.data_mean is None:
        config.sde.data_mean = round(float(stats["mean"]), 4)
    else:
        kept.append(f"sde.data_mean = {config.sde.data_mean}")
    config.save(args.out)

    if kept:
        # Said out loud, because a value carried over from a base config was not
        # measured against *these* shards -- and after a change to out_size or
        # pool_factor the statistics it came from no longer exist.
        print("\nkept from the base config rather than measured:")
        for line in kept:
            print(f"  {line}")
        print("  clear these fields (or drop --base-config) to re-measure")

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
    margin = config.energy.loss_margin
    size = config.patch.out_size
    real = config.real_context(size)
    print(
        f"\nR = {config.energy.receptive_radius} "
        f"({config.energy.n_branches} branch(es), {config.energy.n_layers} "
        f"layers), loss crop 2R = {margin} px per side"
    )
    print(f"  out_size {size} trains on all {size} px; the loader is fed "
          f"{size + 2 * margin}")
    print(f"  of the {margin} px context per side, {min(real, margin):.0f} is "
          f"real sky and {max(margin - real, 0):.0f} is reflected "
          f"(centred crop; translation moves it side to side)")
    print(f"  sizes this stamp can serve: {lo} .. {hi}; context is wholly real "
          f"at or below {max(hi - 2 * margin, 0)}")
    for w in config.check_sizes():
        print(f"  WARNING: {w}")

    cl = dataset.correlation_length(args.n_stats)
    margin = config.energy.loss_margin
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


if __name__ == "__main__":
    main()

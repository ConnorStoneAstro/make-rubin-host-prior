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

* ``sky_scatter`` should match ``expected_sky_scatter(softening_sigma, k)``.  If it
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
    estimate_softening, measure_transform,
    expected_sky_scatter,
    pool_shards,
    reach_advice,
    suggest_n_steps,
    suggest_sigma_range,
)


#: Where the extraction writes its shards, taken from the extraction config
#: rather than written out again here.  ``extract_dp2_patches.py`` writes
#: ``<out>/shards``, and ``out`` has exactly one definition.
DEFAULT_SHARDS = Path(ExtractionConfig.out) / "shards"
DEFAULT_OUT = "config.json"


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
    p.add_argument("--architecture", default=None,
                   choices=["energy", "ncsnpp", "ncsnpp_energy"],
                   help="which network computes the score. 'energy' is the "
                        "dilated-convolution scalar; 'ncsnpp' is the U-Net, "
                        "which predicts sigma*score directly and is not "
                        "conservative; 'ncsnpp_energy' is the same U-Net read "
                        "as a scalar energy, which is what HMC needs. Omit to "
                        "use the config's own value")
    p.add_argument("--pool-factor", type=int, default=None,
                   help="omit to use the config's own value")
    p.add_argument("--out-size", type=int, default=None,
                   help="pooled training patch size; omit to use the config's "
                        "own value. nominal_crop follows as out_size x "
                        "pool_factor")
    p.add_argument("--min-shift-native", type=int, default=None,
                   help="smallest allowed displacement of the training crop "
                        "from the stamp centre, in native px, measured as a "
                        "radius. The host is cut at the centre, so this frames "
                        "every patch with the host that far out -- which is "
                        "how a transient sits on its host, and is the setting "
                        "for a mock-data model. Omit to use the config's own "
                        "value (0: crops may be centred)")
    p.add_argument("--max-shift-native", type=int, default=None,
                   help="largest allowed displacement per axis, in native px. "
                        "Omit to use the config's own value (null: all the "
                        "room the stamp has, (native_size - out_size x "
                        "pool_factor) / 2)")
    p.add_argument("--sigma-max-rule", default="pairs",
                   choices=["pairs", "per-pixel"],
                   help="how to pick sigma_max when the config does not fix "
                        "it. 'pairs' is the standard rule (Song & Ermon 2020 "
                        "Technique 1): the largest distance between two "
                        "training images, and the default. 'per-pixel' is "
                        "the old max(2 std, per-patch range), which is "
                        "~sqrt(D) too small -- it drops structure above some "
                        "size from the samples -- and is here only to "
                        "reproduce a config written before")
    p.add_argument("--n-pairs", type=int, default=512,
                   help="patches to draw the pairwise distances from; every "
                        "pair of them is used, so this is ~n^2/2 distances. "
                        "Not a config field")
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
    if args.architecture is not None:
        config.architecture = args.architecture
    if args.pool_factor is not None:
        config.patch.pool_factor = args.pool_factor
    if args.out_size is not None:
        config.patch.out_size = args.out_size
    if args.softening_sigma is not None:
        config.transform.softening_sigma = args.softening_sigma
    if args.min_shift_native is not None:
        config.patch.min_shift_native = args.min_shift_native
    if args.max_shift_native is not None:
        config.patch.max_shift_native = args.max_shift_native

    # Measured, like everything below it: the stamps are as big as the
    # extraction made them.  nominal_crop follows from out_size and pool_factor
    # on its own, so there is nothing to keep in step with them here.
    config.patch.native_size = shards.native_size
    if shards.native_size < config.patch.nominal_crop:
        raise SystemExit(
            f"the shards are {shards.native_size} px native, but out_size "
            f"{config.patch.out_size} x pool_factor {config.patch.pool_factor} "
            f"needs {config.patch.nominal_crop}. Lower --out-size, or extract "
            f"larger stamps."
        )
    # The geometry above was assigned straight onto the dataclass, which does
    # not re-run __post_init__, so the check that min_shift_native fits inside
    # the stamp has to be asked for.  Here, before the minutes of I/O below,
    # and not at save time: this is where the numbers came from.
    try:
        config.patch.__post_init__()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    softening_sigma = config.transform.softening_sigma
    # Measured from pooled patches, not derived from the variance plane:
    # coadd pixel noise is correlated, so pooling reduces it by less than
    # pool_factor and the derived value would be badly low.
    pooled, _ = pool_shards(shards, config, n=args.n_stats)
    # Measured only where nothing has been set.  None means "not measured yet";
    # a number means someone chose it, and choosing one is the only way to
    # override the heuristic -- there is deliberately no flag for that.
    kept: list[str] = []
    # Measured together, from the same patches: both describe the pooled sky
    # noise, and a config carrying one without the other is half configured.
    measured = measure_transform(shards, config, pooled)
    if config.transform.softening is None:
        config.transform.softening = measured["softening"]
    else:
        kept.append(f"transform.softening = {config.transform.softening}")
    if config.transform.pooled_noise_ratio is None:
        config.transform.pooled_noise_ratio = round(
            measured["pooled_noise_ratio"], 4)
    else:
        kept.append(
            f"transform.pooled_noise_ratio = {config.transform.pooled_noise_ratio}")

    # One scale for every band, so the band counts are reported and nothing
    # more: a band with no patches no longer leaves a hole in the transform.
    # It is still worth seeing, because a band missing entirely usually means
    # the extraction stopped at its target before reaching it.
    counts = shards.band_counts()
    print("patches per band: " + ", ".join(f"{b}={counts[b]}" for b in shards.bands))
    absent = [b for b in shards.bands if counts[b] == 0]
    if absent:
        print(
            f"  NOTE: no patches in {absent}. Check the manifest's "
            f"rejection_counts -- a band missing entirely is usually a run that "
            f"stopped at its target before reaching it, or coadds that do not "
            f"exist for those tracts."
        )

    transform = LogFluxTransform.from_config(config.transform)
    # Never cached: this script reads a few hundred patches for the statistics
    # and exits, so loading the whole set would be minutes of I/O to throw away.
    dataset = PatchDataset.from_shards(shards, config, transform, in_memory=False)
    stats = dataset.stats(args.n_stats)
    pairs = dataset.pair_distances(args.n_pairs)
    sigma_min, sigma_max = suggest_sigma_range(stats, pairs,
                                               rule=args.sigma_max_rule)
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
    if config.sde.data_std is None:
        config.sde.data_std = round(float(stats["std"]), 4)
    else:
        kept.append(f"sde.data_std = {config.sde.data_std}")
    if config.sde.data_mean is None:
        config.sde.data_mean = round(float(stats["mean"]), 4)
    else:
        kept.append(f"sde.data_mean = {config.sde.data_mean}")
    # Computed here, printed far below: the JSON summary goes out before the
    # human-readable report and both quote it.
    ladder = suggest_n_steps(pairs["dim"], config.sde.sigma_min,
                             config.sde.sigma_max)
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
                "pair_distances": pairs,
                "min_sampler_steps": ladder,
                "softening_nJy": config.transform.softening,
                "sigma_min": config.sde.sigma_min,
                "sigma_max": config.sde.sigma_max,
                "data_std": config.sde.data_std,
                "pooled_noise_ratio": config.transform.pooled_noise_ratio,
            },
            indent=2,
        )
    )

    # The correlation length, measured on the pooled log-space patches the model
    # actually sees.  This is the authoritative version -- the one in the
    # extraction summary is native-resolution flux and is contaminated by the PSF.
    t = LogFluxTransform.from_config(config.transform)
    tc = config.transform
    ss = tc.softening_sigma
    # The *typical* floor: k at the midpoint of its per-stamp draw, times the
    # pooled sky noise, which `softening / softening_sigma` recovers.  Every
    # number below is quoted for that, because quoting the unfloored ones for a
    # floored dataset is how the transform figure came to be wrong.
    k_mid = 0.5 * (tc.floor_sigma_lo + tc.floor_sigma_hi) if tc.floors_enabled else 0.0
    floor = k_mid * (t.softening / ss) if (k_mid and ss) else 0.0
    print(f"\nlog transform:  x = log(c + s*softplus((f-c)/s)),  model map f = exp(x)")
    print(f"  softening s = {t.softening:.1f} nJy ({ss:.2f} x pooled sky noise)"
          f" -- how sharp the knee is")
    if floor:
        print(f"  sky floor   c = {floor:.1f} nJy (k = {tc.floor_sigma_lo:.2f}"
              f"..{tc.floor_sigma_hi:.2f} x each cutout's own pooled sky noise,"
              f" drawn per stamp)")
    else:
        print(f"  sky floor   none (floor_sigma_hi = 0): the pre-floor transform")
    print(f"  bright flux passes through as log(f), the same x in every band")
    print(f"  sky sits at log(c + s*log2) = {float(t.sky_level_at(floor)):.2f} "
          f"for a typical cutout; each one floors at its own level")
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
        + ("-> floored, as anything below c is" if floor
           else "-> representable (softplus has no floor)")
    )
    predicted = expected_sky_scatter(ss, k_mid)
    print(
        f"  sky scatter in x: {stats['sky_scatter']:.3f} measured vs "
        f"{predicted:.3f} predicted"
        + (f" at k = {k_mid:.2f}" if k_mid else " (unfloored)")
    )
    if not 0.5 * predicted < stats["sky_scatter"] < 2.0 * predicted:
        print(
            "  WARNING: measured sky scatter is far from prediction -- the "
            + ("floor or the softening scale is probably wrong." if floor
               else "softening scale is probably wrong.")
        )

    # -- sigma_max, the standard way -------------------------------------
    #
    # Reported whichever rule was used, because the two differ by sqrt(D) and
    # the only way to see that a config is on the wrong side of it is to have
    # both numbers side by side.
    per_pixel = float(max(2.0 * stats["std"], stats["per_patch_range_p99"]))
    root_d = pairs["dim"] ** 0.5
    print(f"\npairwise distances between training images "
          f"({pairs['n_pairs']:,} pairs from {pairs['n_patches']} patches, "
          f"D = {pairs['dim']:,} px):")
    print(f"  ||x_i - x_j||_2   mean {pairs['mean']:.1f}  "
          f"std {pairs['std']:.1f}  max {pairs['max']:.1f}")
    print(f"  per pixel (/{root_d:.0f})   mean "
          f"{pairs['mean_per_pixel']:.3f}  max {pairs['max_per_pixel']:.3f}"
          f"   [compare data_std = {config.sde.data_std}]")
    print(f"  median {pairs['percentiles']['50']:.1f}, "
          f"p90 {pairs['percentiles']['90']:.1f}, "
          f"p99 {pairs['percentiles']['99']:.1f}, min {pairs['min']:.1f}")
    print(f"  loader range: x in [{pairs['x_min']:.2f}, {pairs['x_max']:.2f}]")
    print(f"  sigma_max by the standard rule (Song & Ermon 2020, Technique 1:")
    print(f"    the largest pairwise distance) = {pairs['max']:.1f}")
    print(f"  sigma_max by the old per-pixel heuristic   = {per_pixel:.2f}  "
          f"(ratio {pairs['max'] / max(per_pixel, 1e-9):.0f}x)")
    print(f"  in use: {config.sde.sigma_max}")
    print(f"  steps to cross it (Song & Ermon Technique 2: adjacent levels "
          f"must overlap)")
    print(f"    gamma = {ladder['gamma']:.4f} per level  ->  "
          f"n_steps >= {ladder['n_steps']} from sigma_min "
          f"{config.sde.sigma_min} to sigma_max {config.sde.sigma_max}")
    print(f"    the same sqrt(D) the other way round: high dimension makes "
          f"the shells thin, so the ladder has to be fine")
    if config.sde.sigma_max and config.sde.sigma_max < 0.5 * pairs["max"]:
        print(
            "  WARNING: sigma_max is below half the largest pairwise "
            "distance. The t=1 marginal then cannot carry a sample from one "
            "training image to another, so which mode the reverse process "
            "lands in is decided by the initial draw rather than by the "
            "score -- which looks exactly like a model with no large-scale "
            "structure. Clear sde.sigma_max to re-measure."
        )

    from rubin_host_prior import geometry

    lo, hi = config.usable_size_range()
    size = config.patch.out_size
    print(f"\narchitecture: {config.architecture}")
    if config.architecture == "energy":
        margin = config.energy.loss_margin
        R = config.energy.receptive_radius
        pad = geometry.padding_fraction(size, config.energy.dilations,
                                        config.energy.kernel_size)
        print(
            f"  R = {R} ({config.energy.n_branches} branch(es), "
            f"{config.energy.n_layers} layers), score reach 2R = {2 * R} px"
        )
        print(f"  grid {size} ({size * config.patch.pool_factor} of "
              f"{config.patch.native_size} native px), loss on "
              f"{size - 2 * margin} px")
        print(f"  same-mode zero padding: {100 * pad:.0f}% of the mean "
              f"receptive field is padding, and the input is centred on "
              f"{config.input_offset:.2f} so those zeros sit at the sky")
    else:
        cfg = config.ncsnpp
        if config.architecture == "ncsnpp_energy":
            print(f"  scalar energy by '{cfg.energy_form}'; score is -grad_x E, "
                  f"so a step is a gradient of a gradient (2-3x, halve the batch)")
        print(f"  nf {cfg.nf}, ch_mult {tuple(cfg.ch_mult)}, "
              f"{cfg.num_blocks} blocks/level -> resolutions "
              f"{[size // 2 ** i for i in range(cfg.n_levels)]}")
        print(f"  grid {size} ({size * config.patch.pool_factor} of "
              f"{config.patch.native_size} native px); the coarsest level sees "
              f"the whole scene, so there is no reach to compare with xi")
        print(f"  input centred on {config.input_offset:.2f} so the "
              f"convolutions' zero padding sits at the sky")
    print(f"  sizes this stamp can serve: {lo} .. {hi}; crop shift "
          f"{config.patch.shift_note}")
    for w in config.check_sizes():
        print(f"  WARNING: {w}")

    from rubin_host_prior import plots

    head = plots.schedule_headroom(
        dataset.validation_batch(min(args.n_stats, len(dataset))),
        config.sde.sigma_max, (1, 2, 4, 8, 16, 32))
    print(f"\nsigma_max = {config.sde.sigma_max:.4g} against the scales it has "
          f"to erase")
    print("  sigma_vis = sqrt(P) for the band. A mode of power P is still")
    print("  visible at noise sigma whenever P > sigma^2, so the schedule must")
    print("  start above the LARGEST sigma_vis -- not above the per-pixel std,")
    print("  which is the mean of P over modes and says nothing about the few")
    print("  that hold a red field's variance. 'floor' is the band ratio a")
    print("  PERFECT score would produce from this sigma_max: 1/sqrt(1+P/s^2).")
    print(f"\n  {'scale':>7} {'sigma_vis':>10} {'headroom':>9} {'floor':>7}")
    for i, band in enumerate(head["band"]):
        flag = "  <-- too small" if head["headroom"][i] < 3.0 else ""
        print(f"  {band:>7} {head['sigma_visible'][i]:>10.4g} "
              f"{head['headroom'][i]:>9.1f} {head['pflow_ratio'][i]:>7.3f}{flag}")
    worst = min(head["pflow_ratio"])
    if worst < 0.9:
        band = head["band"][head["pflow_ratio"].index(worst)]
        hurt = [b for b, f in zip(head["band"], head["pflow_ratio"]) if f < 0.9]
        print(f"\n  WARNING: even a perfect score would reach only "
              f"{worst:.2f} of the real power at {band} px. Raise "
              f"sde.sigma_max to at least 3x the largest sigma_vis "
              f"({3 * max(head['sigma_visible']):.0f}) and re-measure.")
        print(f"  What this looks like in the samples: structure at "
              f"{', '.join(hurt)} px comes out at a fraction of its true "
              f"amplitude, so **objects above that size are missing from "
              f"draws although they are plainly in the training data**. The "
              f"model is not at fault and more training will not fix it -- "
              f"prior_sample starts from white noise of variance sigma_max^2, "
              f"the true t=1 marginal has variance P + sigma_max^2, and the "
              f"shortfall is the initial condition.")

    cl = dataset.correlation_length(args.n_stats)
    print(f"\ncorrelation length (pooled, log space, over {cl['n_patches']} patches)")
    print(f"  profile: " + " ".join(f"{v:.2f}" for v in cl["profile"][:10]))
    print(
        f"  {cl['noise_fraction']:.0%} of the variance is the zero-lag noise "
        f"delta (excluded from xi)"
    )
    if config.architecture == "energy":
        print(f"  {reach_advice(cl['xi'], 2 * config.energy.receptive_radius)}")
    else:
        print(f"  xi = {cl['xi']:.1f} px against a {size} px grid the U-Net "
              f"sees all of at its coarsest level")
    if cl["truncated"]:
        print(
            "  WARNING: the patches never decorrelate within their own size, so "
            "xi is a lower bound. Extract larger patches to measure it."
        )


if __name__ == "__main__":
    main()

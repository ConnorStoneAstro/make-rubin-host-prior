#!/usr/bin/env python
"""Watch a sample come out of the noise: the sampler, with a recorder on it.

    python scripts/reverse_trajectory.py --checkpoint runs/r/final

Rows are samples, columns are the trained schedule's own noise levels
*descending*, each on its own stretch, so the rightmost column is the finished
sample.  The states are the sampler's real intermediates -- ``pflow_trajectory``
and ``pflow_sample`` share one integration, so the last column is exactly what
sampling produces for this key.

Beneath them is the check that costs nothing and catches a diverging sampler:
the forward marginal at ``sigma`` has width ``sqrt(var(x) + sigma^2)``, so a
trajectory whose spread departs from that curve is not tracking the
distribution it is meant to be reversing, whatever the pictures look like.
Pass ``--shards`` and the data's own variance is measured and drawn with it.

**This is the companion to ``forward_diffusion.py``**: the same ``--n-sigma``
gives both figures the same noise levels, and the column where they stop
resembling each other is the sigma range to suspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jax

from rubin_host_prior import plots
from rubin_host_prior.data import LogFluxTransform, PatchDataset, ShardSet
from rubin_host_prior.diffusion import VESDE
from rubin_host_prior.training import load_checkpoint


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="diagnostics")
    p.add_argument("--shards", default=None,
                   help="optional but worth it: real patches for the two "
                        "bottom panels to compare the samples against, drawn "
                        "exactly as forward_diffusion.py draws them")
    p.add_argument("--scales", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32],
                   help="spatial scales, in pooled pixels, for the power panel")
    p.add_argument("--n", type=int, default=4, help="samples, one per row")
    p.add_argument("--n-sigma", type=int, default=8, help="columns")
    p.add_argument("--steps", type=int, default=None,
                   help="default: the config's train.sample_steps")
    p.add_argument("--size", type=int, default=None,
                   help="default: the config's out_size, which is the grid the "
                        "model was trained on and the only one it is defined on")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    model, config, step = load_checkpoint(args.checkpoint, which=args.weights)
    sde = VESDE.from_config(config.sde)
    size = args.size or config.patch.out_size
    steps = args.steps or config.train.sample_steps
    if size != config.patch.out_size:
        print(f"  WARNING: sampling at {size} but the model was trained on "
              f"{config.patch.out_size}; zero padding makes the training grid "
              f"part of the operator, so this is a different prior.")

    reference = None
    if args.shards:
        shards = ShardSet.from_dir(args.shards)
        dataset = PatchDataset.from_shards(
            shards, config, LogFluxTransform.from_config(config.transform),
            in_memory=False)
        # Augmented, and that is not cosmetic: the model was trained on
        # translated crops, so the distribution it is matching is the one where
        # the host can sit anywhere and can be cut by the frame.  A centred
        # reference has 16% more power at 32 px than the training distribution
        # does, and charging the samples for that is charging them for the
        # loader's convention.  More patches than the sample count, because the
        # augmentation draw adds its own scatter -- ~4% at the largest band.
        reference = plots.forward_patches(dataset, 64, augment=True,
                                          seed=args.seed)

    # The same lines forward_diffusion.py prints, from the same file, so the
    # two can be put next to each other and checked rather than assumed.
    p = config.patch
    print(f"config      {Path(args.checkpoint) / 'config.json'}")
    print(f"schedule    sigma {sde.sigma_min:.5g} .. {sde.sigma_max:.5g}, "
          f"data mean {sde.data_mean:.4g}")
    print(f"transform   softening {config.transform.softening:.4g} nJy "
          f"(sky at {config.input_offset:.4g} in x)")
    print(f"patches     {p.out_size} px grid, pool {p.pool_factor}, "
          f"native {p.native_size}")
    print(f"checkpoint step {step}; {args.n} trajectories of {steps} steps "
          f"at {size}x{size}")
    fig, path = plots.plot_reverse_trajectory(
        model, sde, size, n=args.n, n_sigma=args.n_sigma, n_steps=steps,
        seed=args.seed, reference=reference, scales=tuple(args.scales),
        out=Path(args.out))

    if reference is not None:
        from rubin_host_prior.diffusion import pflow_sample

        final = np.asarray(pflow_sample(
            model, jax.random.key(args.seed),
            (args.n, model.config.in_channels, size, size), sde, n_steps=steps))
        got = plots.scale_visibility(final, tuple(args.scales))
        want = plots.scale_visibility(reference, tuple(args.scales))
        print("\nband power of the samples against real patches, per octave")
        print("1.00 = the right amount of structure at that scale")
        print("(+- is the statistical error; the largest band is only tens of")
        print(" modes, so raise --n if it is wider than the effect)\n")
        print(f"{'scale':>7} {'samples':>10} {'real':>10} {'ratio':>8} "
              f"{'+-':>7} {'modes':>7}")
        for i, sc in enumerate(got["scales"]):
            a, b = got["signal"][i], want["signal"][i]
            err = (got["rel_error"][i] ** 2 + want["rel_error"][i] ** 2) ** 0.5
            print(f"{sc:>7} {a:>10.4g} {b:>10.4g} {a / b:>8.2f} "
                  f"{err * a / b:>7.2f} {got['n_modes'][i]:>7,}")
        ga = plots.band_power(final, tuple(args.scales))
        gb = plots.band_power(reference, tuple(args.scales))
        # Everything coarser than the largest band, which is only a handful of
        # modes and, in a field this red, is not a handful of the variance.
        below_ratio = ga["below_rms"] / gb["below_rms"]
        below_err = (ga["below_rel_error"] ** 2
                     + gb["below_rel_error"] ** 2) ** 0.5
        print(f"{'>' + str(ga['below_scale']):>7} {ga['below_rms']:>10.4g} "
              f"{gb['below_rms']:>10.4g} {below_ratio:>8.2f} "
              f"{below_err * below_ratio:>7.2f} {ga['below_modes']:>7,}")
        print(f"{'all':>7} {ga['total_rms']:>10.4g} {gb['total_rms']:>10.4g} "
              f"{ga['total_rms'] / gb['total_rms']:>8.2f}"
              f"{'':>8} {sum(ga['n_modes']) + ga['below_modes']:>7,}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

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

from rubin_host_prior import plots
from rubin_host_prior.data import LogFluxTransform, PatchDataset, ShardSet
from rubin_host_prior.diffusion import VESDE
from rubin_host_prior.training import load_checkpoint


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="diagnostics")
    p.add_argument("--shards", default=None,
                   help="optional; measures the data variance for the width "
                        "reference in the bottom panel")
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

    data_std = None
    if args.shards:
        shards = ShardSet.from_dir(args.shards)
        dataset = PatchDataset.from_shards(
            shards, config, LogFluxTransform.from_config(config.transform),
            in_memory=False)
        data_std = float(np.std(dataset.validation_batch(
            min(64, len(dataset)))))
        print(f"data std {data_std:.4g} (from {args.shards})")

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
        seed=args.seed, data_std=data_std, out=Path(args.out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

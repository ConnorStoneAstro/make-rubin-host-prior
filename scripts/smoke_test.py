#!/usr/bin/env python
"""End-to-end dry run on synthetic data.  No LSST stack, no cluster, ~3 minutes.

    python scripts/smoke_test.py

It was a minute when the model was eight plain layers and the loader handed over
exactly ``out_size``.  A residual stack, dilations, and a patch carrying ``2R``
of context on every side cost the rest -- at the old defaults this now takes
**12 minutes**, measured, which is long enough that nobody runs it.  So the
defaults came down (300 steps to 200, batch 16 to 8, out_size 32 to 24) rather
than the promise being quietly restated: the loss still falls 0.989 -> 0.968
over the run, so the PASS at the end still means something.

Exercises the whole chain -- synthetic shards, offset estimation, the loader, the
energy net, the loss, training, checkpointing, sampling -- so that porting to
NERSC only has to debug the Butler part.  Run it after any change to the model
or the transform.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

from rubin_host_prior.config import Config, EnergyConfig
from rubin_host_prior.data import (
    LogFluxTransform,
    PatchDataset,
    ShardSet,
    estimate_softening,
    expected_sky_scatter,
    pool_shards,
    reach_advice,
    suggest_sigma_range,
)
from rubin_host_prior.data.synthetic import write_synthetic_shards
from rubin_host_prior.diffusion import VESDE, mean_dsm_loss, sample_scene
from rubin_host_prior.nn import n_parameters
from rubin_host_prior.training import load_checkpoint, train


def wedge(n: int) -> tuple[int, ...]:
    """A doubling series up and back down again, in ``n`` layers.

    The same shape as the project default, scaled to whatever depth the smoke
    test is run at: the ascent reaches and the descent de-grids.  Built rather
    than sliced off ``DEFAULT_DILATIONS`` because a slice of it is a bare
    ascent -- no de-gridding tail, and a reach that grows like ``2^n``, which at
    four layers already wants a stamp three times the size.
    """
    up = tuple(2 ** i for i in range((n + 1) // 2))
    return up + tuple(reversed(up))[n % 2:]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-patches", type=int, default=256)
    p.add_argument("--out-size", type=int, default=24)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--out-sizes", type=int, nargs="+", default=None,
                   help="also train at these sizes, cycled round-robin")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workdir", default=None)
    args = p.parse_args()

    work = Path(args.workdir or tempfile.mkdtemp(prefix="rhp-smoke-"))
    print(f"workdir: {work}\n")

    # Uniform width so the residual skips are possible, and the same
    # rise-and-fall dilation shape as the default, so the run exercises dilation
    # and residual together rather than a plain stack -- this is the rehearsal
    # for NERSC, so the shapes it checks should be the shapes that go there.
    energy = EnergyConfig(
        channels=((32,) * args.n_layers,),
        dilations=(wedge(args.n_layers),),
    )
    # The stamp holds the crop plus slack for translation -- same-mode
    # convolutions need no context border, so the crop is the whole story.
    native = args.out_size * 2 + 32

    t0 = time.time()
    write_synthetic_shards(
        work / "shards",
        n_patches=args.n_patches,
        native_size=native,
        seed=0,
    )
    shards = ShardSet.from_dir(work / "shards")
    print(f"[1] {len(shards)} synthetic patches, native {shards.native_size} "
          f"({time.time() - t0:.1f}s)")

    config = Config()
    config.patch.out_size = args.out_size
    config.patch.pool_factor = 2
    config.patch.native_size = shards.native_size
    if args.out_sizes:
        config.patch = dataclasses.replace(
            config.patch, out_sizes=tuple(args.out_sizes)
        )
    config.energy = energy
    config.train.steps = args.steps
    config.train.batch_size = args.batch_size
    config.train.log_every = max(args.steps // 10, 1)
    config.train.n_checkpoints = 0
    pooled, pooled_bands = pool_shards(shards, config)
    config.transform.softening = estimate_softening(
        pooled, config.transform.softening_sigma
    )
    transform = LogFluxTransform.from_config(config.transform)
    dataset = PatchDataset.from_shards(shards, config, transform)
    stats = dataset.stats(128)
    config.sde.sigma_min, config.sde.sigma_max = suggest_sigma_range(stats)
    # x is absolute log flux, so the data is not centred on zero and the t=1
    # marginal is not either.
    config.sde.data_mean = float(stats["mean"])
    print(f"[2] softening {config.transform.softening:.1f} nJy")
    print(f"    sky_scatter {stats['sky_scatter']:.3f} (expect ~"
          f"{expected_sky_scatter(config.transform.softening_sigma):.2f}), "
          f"deepest {stats['deepest_flux_sigma']:.1f} sigma")
    print(f"    sky level {stats['sky_level']:.2f} (every band)")
    print(f"    sigma range [{config.sde.sigma_min:.4f}, {config.sde.sigma_max:.2f}]"
          f" about data mean {config.sde.data_mean:.2f}")
    cl = dataset.correlation_length(128)
    print(f"    {reach_advice(cl['xi'], 2 * config.energy.receptive_radius)}"
          f"   ({cl['noise_fraction']:.0%} of variance is noise)")

    model = config.build_model(jax.random.key(0))
    print(f"[3] {n_parameters(model):,} parameters; "
          f"training sizes {config.patch.training_sizes}")

    sde = VESDE.from_config(config.sde)
    # Averaged over many batches: a single batch's loss is dominated by its
    # sigma draw and says nothing about whether the model improved.
    before = mean_dsm_loss(
        model, dataset.batches(config.train.batch_size, seed=99), 30,
        jax.random.key(7), sde,
    )
    t0 = time.time()
    model, ema = train(
        model,
        dataset.batches(config.train.batch_size, seed=0),
        config,
        out_dir=work / "run",
        sde=sde,
    )
    after = mean_dsm_loss(
        ema, dataset.batches(config.train.batch_size, seed=99), 30,
        jax.random.key(7), sde,
    )
    print(f"[4] trained {args.steps} steps in {time.time() - t0:.1f}s")
    print(f"    mean loss {before:.4f} -> {after:.4f}  (1.0 = no score learned)")

    reloaded, cfg2, step = load_checkpoint(work / "run" / "final", which="ema")
    print(f"[5] reloaded checkpoint at step {step}, "
          f"{n_parameters(reloaded):,} parameters")

    t0 = time.time()
    x = sample_scene(
        reloaded, jax.random.key(1), out_size=args.out_size, n_samples=4,
        sde=VESDE.from_config(cfg2.sde), n_steps=64,
    )
    x = np.asarray(x)
    print(f"[6] sampled {x.shape} in {time.time() - t0:.1f}s; "
          f"mean {x.mean():+.3f} std {x.std():.3f}")

    ok = after < before and np.all(np.isfinite(x))
    print("\n" + ("PASS" if ok else "FAIL") + f"  (artifacts in {work})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

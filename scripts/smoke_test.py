#!/usr/bin/env python
"""End-to-end dry run on synthetic data.  No LSST stack, no cluster, ~1 minute.

    python scripts/smoke_test.py --steps 300

Exercises the whole chain -- synthetic shards, offset estimation, the loader, the
energy net, the loss, training, checkpointing, sampling -- so that porting to
NERSC only has to debug the Butler part.  Run it after any change to the model
or the transform.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

from rubin_host_prior import geometry
from rubin_host_prior.config import Config, EnergyConfig
from rubin_host_prior.data import (
    LogFluxTransform,
    PatchDataset,
    ShardSet,
    estimate_band_offsets,
    suggest_sigma_range,
)
from rubin_host_prior.data.synthetic import write_synthetic_shards
from rubin_host_prior.diffusion import VESDE, mean_dsm_loss, sample_interior
from rubin_host_prior.nn import ConvEnergyNet, n_parameters
from rubin_host_prior.training import load_checkpoint, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--n-patches", type=int, default=256)
    p.add_argument("--out-size", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workdir", default=None)
    args = p.parse_args()

    work = Path(args.workdir or tempfile.mkdtemp(prefix="rhp-smoke-"))
    print(f"workdir: {work}\n")

    t0 = time.time()
    write_synthetic_shards(
        work / "shards",
        n_patches=args.n_patches,
        native_size=args.out_size * 3 + 32,
        seed=0,
    )
    shards = ShardSet.from_dir(work / "shards")
    print(f"[1] {len(shards)} synthetic patches, native {shards.native_size} "
          f"({time.time() - t0:.1f}s)")

    config = Config()
    config.patch.out_size = args.out_size
    config.patch.pool_factor = 3
    config.patch.nominal_crop = args.out_size * 3
    config.patch.native_size = shards.native_size
    config.energy = EnergyConfig(
        channels=tuple([16, 24, 32, 32][min(i, 3)] for i in range(args.n_layers))
    )
    config.train.steps = args.steps
    config.train.batch_size = args.batch_size
    config.train.log_every = max(args.steps // 10, 1)
    config.train.ckpt_every = 0
    config.transform.band_offsets = estimate_band_offsets(
        shards.load("variance"), shards.meta["band_idx"], 3, config.transform.k_sigma
    )
    transform = LogFluxTransform.from_config(config.transform)
    dataset = PatchDataset.from_shards(shards, config, transform)
    stats = dataset.stats(128)
    config.sde.sigma_min, config.sde.sigma_max = suggest_sigma_range(stats)
    print(f"[2] offsets {({k: round(v, 1) for k, v in config.transform.band_offsets.items()})}")
    print(f"    sky_scatter {stats['sky_scatter']:.3f} (expect "
          f"~{1 / config.transform.k_sigma:.2f}), clipped {stats['clipped_fraction']:.2e}")
    print(f"    sigma range [{config.sde.sigma_min:.4f}, {config.sde.sigma_max:.2f}]")

    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    print(f"[3] {geometry.describe(args.out_size, model.n_layers)}")
    print(f"    {n_parameters(model):,} parameters")

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
    x = sample_interior(
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

#!/usr/bin/env python
"""Train the energy-based score model.

    python scripts/train.py --shards data/ecdfs_r/shards --config config.json \
        --out runs/ecdfs_r

Checks the patch/layer geometry before starting, since a too-small patch leaves
no interior for the loss and that is better caught now than 10 000 steps in.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging

import jax

from rubin_host_prior import geometry
from rubin_host_prior.config import Config
from rubin_host_prior.data import LogFluxTransform, PatchDataset, ShardSet
from rubin_host_prior.diffusion import VESDE
from rubin_host_prior.nn import ConvEnergyNet, n_parameters
from rubin_host_prior.training import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", default=None, help="directory of *.h5 shards")
    p.add_argument("--pooled-cache", default=None, help="use a pooled cache instead")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-layers", type=int, default=None,
                   help="override the layer count, keeping the widths pattern")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-size", type=int, default=32)
    p.add_argument("--max-in-memory-gb", type=float, default=16.0)
    p.add_argument("--verbose", "-v", action="count", default=1)
    args = p.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    if not args.shards and not args.pooled_cache:
        p.error("need --shards or --pooled-cache")

    config = Config.load(args.config)
    if args.steps:
        config.train.steps = args.steps
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    if args.n_layers:
        base = config.energy.channels
        config.energy = dataclasses.replace(
            config.energy,
            channels=tuple(
                base[min(i, len(base) - 1)] for i in range(args.n_layers)
            ),
        )

    transform = LogFluxTransform.from_config(config.transform)
    if args.pooled_cache:
        dataset = PatchDataset.from_pooled_cache(
            args.pooled_cache, config, transform
        )
    else:
        shards = ShardSet.from_dir(args.shards)
        dataset = PatchDataset.from_shards(
            shards, config, transform, max_in_memory_gb=args.max_in_memory_gb
        )

    model = ConvEnergyNet(config.energy, key=jax.random.key(config.train.seed))
    print(geometry.describe(config.patch.out_size, model.n_layers,
                            config.energy.kernel_size))
    print(f"{n_parameters(model):,} parameters | {len(dataset):,} patches "
          f"| loader mode {dataset.mode}")
    print(json.dumps(dataset.stats(min(256, len(dataset))), indent=2))

    train(
        model,
        dataset.batches(config.train.batch_size, seed=config.train.seed),
        config,
        out_dir=args.out,
        sde=VESDE.from_config(config.sde),
        eval_batch=dataset.validation_batch(args.eval_size),
        eval_every=args.eval_every,
        on_log=lambda r: print(json.dumps(r)) if "event" not in r else None,
    )


if __name__ == "__main__":
    main()

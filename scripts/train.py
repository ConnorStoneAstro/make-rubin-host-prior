#!/usr/bin/env python
"""Train the energy-based score model.

    python scripts/train.py --shards data/ecdfs_r/shards --config config.json \
        --out runs/ecdfs_r

Checks the patch/layer geometry before starting, since a too-small patch leaves
no interior for the loss and that is better caught now than 10 000 steps in.

Writes into ``--out``:

    log.jsonl                  one line per log step, eval and checkpoint
    checkpoints/step-XXXXXXXX/ the weights at each of ``n_checkpoints`` points
    samples/step-XXXXXXXX.png  an n x n grid drawn from the EMA model there
    latest/                    the newest checkpoint, with the optimiser state
    final/                     the end of the run

The sample grids are in the same log space and the same style as
``training_batch.png`` from ``diagnose.py``, so the two can be compared directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging

import jax

from rubin_host_prior.config import Config
from rubin_host_prior.data import (LogFluxTransform, PatchDataset, ShardSet,
                                   context_advice)
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
    p.add_argument("--out-sizes", type=int, nargs="+", default=None,
                   help="train on several patch sizes, cycled round-robin across "
                        "batches; larger patches spend less of themselves on the "
                        "cropped border")
    p.add_argument("--n-checkpoints", type=int, default=None,
                   help="checkpoints spread evenly over the run (default: from "
                        "the config, 10). Each keeps the weights and writes a "
                        "grid of samples; 0 for none")
    p.add_argument("--n-samples", type=int, default=None,
                   help="samples drawn from the EMA model at each checkpoint "
                        "and written as a square grid; 0 to skip sampling")
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
    # ``is not None``, not truthiness: 0 is a meaningful value for both of these
    # and means "none", which is exactly what a plain `if` would discard.
    if args.n_checkpoints is not None:
        config.train.n_checkpoints = args.n_checkpoints
    if args.n_samples is not None:
        config.train.n_samples = args.n_samples
    if args.n_layers:
        base = config.energy.channels
        config.energy = dataclasses.replace(
            config.energy,
            channels=tuple(
                base[min(i, len(base) - 1)] for i in range(args.n_layers)
            ),
        )

    if args.out_sizes:
        config.patch = dataclasses.replace(
            config.patch, out_sizes=tuple(args.out_sizes)
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
    print(f"{n_parameters(model):,} parameters | {len(dataset):,} patches "
          f"| loader mode {dataset.mode}")
    print(json.dumps(dataset.stats(min(256, len(dataset))), indent=2))
    cl = dataset.correlation_length(min(256, len(dataset)))
    print(context_advice(cl["xi"], model.loss_margin))
    print()
    # train() prints the full valid-convolution geometry, including exactly how
    # much of each patch the loss crop discards.

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

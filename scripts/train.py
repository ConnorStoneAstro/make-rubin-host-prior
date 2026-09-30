#!/usr/bin/env python
"""Train the energy-based score model.

    python scripts/train.py --shards data/ecdfs_r/shards --config config.json \
        --out runs/ecdfs_r

Checks the patch/layer geometry before starting, since a too-small patch leaves
no interior for the loss and that is better caught now than 10 000 steps in.

Data-parallel across every GPU JAX can see unless ``--devices`` says otherwise.
``--batch-size`` is the global batch split across them, so more devices make the
same run faster rather than changing it.

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
from pathlib import Path

import jax

from rubin_host_prior.config import Config
from rubin_host_prior.selection import ExtractionConfig
from rubin_host_prior.data import (LogFluxTransform, PatchDataset, ShardSet,
                                   reach_advice)
from rubin_host_prior.diffusion import VESDE
from rubin_host_prior.nn import n_parameters
from rubin_host_prior.training import train


#: Where the extraction writes its shards, from the extraction config rather
#: than written down a second time.  Same default as prepare_config.py.
DEFAULT_SHARDS = Path(ExtractionConfig.out) / "shards"


def parser() -> argparse.ArgumentParser:
    """Built separately so a test can ask what a flag defaults to.

    Every flag that names a config field defaults to None and is applied only
    when given.  ``config.py`` holds the defaults for the whole project; a
    number written here as well would override the config on every run, passed
    or not, which is how ``prepare_config.py`` spent a while resetting
    ``out_size`` to 64 whatever the config said.

    Two kinds of flag are not config fields and do carry values:
    ``--max-in-memory-gb`` and ``--devices`` describe the machine this run
    happens to be on rather than the run, and ``--shards``, ``--config``,
    ``--out`` and ``--resume`` say where to read and write.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", default=str(DEFAULT_SHARDS),
                   help=f"directory of *.h5 shards (default: {DEFAULT_SHARDS})")
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
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="continue a run. Bare --resume picks <out>/latest if it "
                        "exists and starts fresh if it does not, which is what a "
                        "chunked scheduler job wants: the same command works for "
                        "the first chunk and every one after it")
    p.add_argument("--devices", type=int, default=None,
                   help="data-parallel devices (default: every GPU JAX can "
                        "see). The parameters are replicated and --batch-size "
                        "is the global batch split across them, so this makes "
                        "the same run faster rather than changing it; 1 for "
                        "the single-device path")
    p.add_argument("--eval-every", type=int, default=None,
                   help="steps between validation passes; omit to use the "
                        "config's own value, 0 to switch it off")
    p.add_argument("--eval-size", type=int, default=None,
                   help="patches in the validation batch; omit to use the "
                        "config's own value")
    p.add_argument("--max-in-memory-gb", type=float, default=16.0)
    p.add_argument("--verbose", "-v", action="count", default=1)
    return p


def main() -> None:
    args = parser().parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s: %(message)s",
    )
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
    if args.eval_every is not None:
        config.train.eval_every = args.eval_every
    if args.eval_size is not None:
        config.train.eval_size = args.eval_size
    if args.n_layers:
        # The first branch only: it is the one whose depth is a free choice.
        # A long-range branch's layer count is set by the reach it has to cover,
        # so resizing it here would silently change R and the loss crop.
        base = config.energy.channels[0]
        resized = tuple(base[min(i, len(base) - 1)] for i in range(args.n_layers))
        config.energy = dataclasses.replace(
            config.energy,
            channels=(resized,) + config.energy.channels[1:],
            dilations=((1,) * args.n_layers,) + config.energy.dilations[1:],
        )

    if args.out_sizes:
        config.patch = dataclasses.replace(
            config.patch, out_sizes=tuple(args.out_sizes)
        )

    # "auto" means "carry on if there is anything to carry on from".  A runner
    # script submitting chunk after chunk can then use one command line.
    resume = args.resume
    if resume == "auto":
        latest = Path(args.out) / "latest"
        resume = str(latest) if (latest / "opt_state.eqx").exists() else None
        print(f"--resume auto: {'continuing from ' + resume if resume else
                                'nothing to resume from, starting fresh'}")

    transform = LogFluxTransform.from_config(config.transform)
    shards = ShardSet.from_dir(args.shards)
    dataset = PatchDataset.from_shards(
        shards, config, transform, max_in_memory_gb=args.max_in_memory_gb
    )

    model = config.build_model(jax.random.key(config.train.seed))
    print(f"{n_parameters(model):,} parameters | {len(dataset):,} patches")
    print(dataset.storage_note(args.max_in_memory_gb))
    print(json.dumps(dataset.stats(min(256, len(dataset))), indent=2))
    cl = dataset.correlation_length(min(256, len(dataset)))
    print(reach_advice(cl["xi"], 2 * model.receptive_radius))
    print()
    # train() prints the full valid-convolution geometry, including exactly how
    # much of each patch the loss crop discards.

    # The batch stream is seeded from where this chunk starts, not from the
    # config alone: an iterator cannot be fast-forwarded, so a resumed run given
    # the same seed would replay the exact sequence of batches the previous
    # chunk already trained on.
    start = 0
    if resume:
        start = json.loads((Path(resume) / "state.json").read_text())["step"]

    train(
        model,
        dataset.batches(config.train.batch_size,
                        seed=config.train.seed + start),
        config,
        out_dir=args.out,
        sde=VESDE.from_config(config.sde),
        eval_batch=dataset.validation_batch(config.train.eval_size),
        on_log=lambda r: print(json.dumps(r)) if "event" not in r else None,
        resume=resume,
        n_devices=args.devices,
    )


if __name__ == "__main__":
    main()

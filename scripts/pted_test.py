#!/usr/bin/env python
"""Is the trained prior sampling the training distribution?  A p-value.

    python scripts/pted_test.py --run runs/ncsnpp-energy/final --shards data/hosts

Draws patches from the loader and scenes from the trained model and compares
them with PTED, a permutation test on the energy distance that makes no
distributional assumptions -- which is what a 16384-dimensional galaxy scene
requires.  **The null is "same distribution", so a small p-value means the
model is wrong.**

The two sample sizes are deliberately lopsided.  Data patches are nearly free,
model draws cost an ODE solve each, and the test's sensitivity depends on both
-- so the default takes 16384 of the first and 512 of the second.
``--n-model`` is the knob that decides how hard the test looks; raise it when
a run comes back "not rejected" and you want that to mean something.

The test runs **exact** by default, on the GPU that drew the samples: PTED has
a JAX backend, so both samples stay on device and the full (n, n) distance
matrix is 1.1 GB of float32 and ~4.7 Tflop at the default sizes.  That is a
GPU's natural work and a CPU's bad afternoon -- measured, the same test is of
order an hour on CPU -- so on a machine without one, pass ``--n-landmarks``.

Every run also does the control: the data sample split in half and tested
against itself, several times, which must not reject.  It is cheap next to
the sampling and it is the difference between a p-value you can act on and a
number.  See ``rubin_host_prior.twosample`` for what each piece is for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubin_host_prior import twosample
from rubin_host_prior.config import Config
from rubin_host_prior.data import LogFluxTransform, PatchDataset, ShardSet
from rubin_host_prior.inference import load_prior


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="checkpoint directory, as load_prior takes it")
    p.add_argument("--shards", default="data/hosts",
                   help="directory of *.h5 shards the model was trained on")
    p.add_argument("--config", default=None,
                   help="config JSON for the loader. Defaults to config.json "
                        "beside the checkpoint, which is the one the run used "
                        "-- a different one compares against a different "
                        "distribution and the test will reject for that")
    p.add_argument("--which", default="ema", choices=["ema", "model"],
                   help="which weights; ema is what you would deploy")
    p.add_argument("--n-data", type=int, default=16384,
                   help="training patches to draw (cheap)")
    p.add_argument("--n-model", type=int, default=512,
                   help="model draws (an ODE solve each). This is what sets "
                        "how small a difference the test can see")
    p.add_argument("--n-steps", type=int, default=256,
                   help="sampler steps per draw. Check this against the "
                        "n_steps floor prepare_config reports: too few and "
                        "you are testing the sampler, not the model")
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--n-landmarks", type=int, default=None,
                   help="landmark points for the energy distance. Omit for "
                        "the exact full-matrix test, which is the default and "
                        "what a GPU is for; pass an int to trade sensitivity "
                        "for cost on a machine that cannot hold an (n, n) "
                        "matrix or wait O(n^2 D) for it")
    p.add_argument("--backend", default="jax", choices=["jax", "numpy"],
                   help="where the distance matrix is built. 'jax' keeps both "
                        "samples on whatever device drew them, which is the "
                        "point of having a GPU attached; 'numpy' is scipy's "
                        "cdist in float64")
    p.add_argument("--control-repeats", type=int, default=5,
                   help="data-vs-data splits; 0 to skip, which is a bad idea")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="rejection threshold, chosen before you look")
    p.add_argument("--no-augment", action="store_true",
                   help="draw the data WITHOUT augmentation. Almost always "
                        "wrong: the model learnt the augmented distribution, "
                        "so this rejects on the augmentation alone")
    p.add_argument("--chunk", type=int, default=64,
                   help="model draws per sampler call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the result as JSON here")
    return p


def main() -> None:
    args = parser().parse_args()

    run = Path(args.run)
    cfg_path = Path(args.config) if args.config else run / "config.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"no config at {cfg_path}. The loader has to be built from the "
            f"config the run trained with -- pass --config explicitly."
        )
    config = Config.load(cfg_path)
    shards = ShardSet.from_dir(args.shards)
    dataset = PatchDataset.from_shards(
        shards, config, LogFluxTransform.from_config(config.transform),
        in_memory=False)
    prior = load_prior(run, which=args.which)

    if prior.size != config.patch.out_size:
        raise SystemExit(
            f"the prior is {prior.size} px and the config's loader yields "
            f"{config.patch.out_size} px. The prior is size-locked, so these "
            f"are not comparable."
        )

    print(prior.summary())
    import jax

    print(f"\njax devices: {jax.devices()}")
    print(f"drawing {min(args.n_data, len(dataset))} patches from "
          f"{len(dataset)} available"
          + ("" if not args.no_augment else "  (augmentation OFF)"))
    data = twosample.data_sample(dataset, args.n_data, seed=args.seed,
                                 augment=not args.no_augment)

    landmarks = args.n_landmarks
    cost = twosample.estimate_cost(len(data), args.n_model,
                                   prior.size ** 2, landmarks)
    print(f"  {'exact' if cost['exact'] else 'landmark'} test: "
          f"{cost['pooled']} x {cost['columns']} distance matrix "
          f"({cost['matrix_gb']:.2f} GB) over {cost['samples_gb']:.2f} GB of "
          f"samples, {cost['flops'] / 1e12:.1f} Tflop")

    # The control first, on purpose.  It needs no model draws and builds very
    # nearly the same distance matrix, so if this is going to run out of
    # memory it does so before the sampling budget is spent rather than after.
    ctrl = None
    if args.control_repeats > 0:
        print(f"\ncontrol: {args.control_repeats} data-vs-data splits "
              f"(before sampling, so an OOM costs nothing)")
        ctrl = twosample.control(data, permutations=args.permutations,
                                 n_landmarks=landmarks, seed=args.seed,
                                 repeats=args.control_repeats,
                                 backend=args.backend)
        ps = " ".join(f"{v:.3g}" for v in ctrl["p_values"])
        print(f"  p = {ps}   ({ctrl['seconds']:.1f}s)")

    print(f"\ndrawing {args.n_model} scenes from the model "
          f"({args.n_steps} sampler steps each)")
    model = twosample.model_sample(prior, args.n_model, seed=args.seed,
                                   chunk=args.chunk, progress=True,
                                   n_steps=args.n_steps)

    print("comparing")
    result = twosample.compare(data, model, permutations=args.permutations,
                               n_landmarks=landmarks, seed=args.seed,
                               backend=args.backend)

    print()
    print(twosample.report(result, ctrl, alpha=args.alpha))

    if args.out:
        payload = {"run": str(run), "which": args.which,
                   "backend": args.backend, "cost": cost,
                   "n_steps": args.n_steps, "augmented": not args.no_augment,
                   "alpha": args.alpha, "result": result, "control": ctrl}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

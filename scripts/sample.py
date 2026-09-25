#!/usr/bin/env python
"""Draw scenes from a trained prior and save them as a PNG mosaic plus a .npy.

    python scripts/sample.py --checkpoint runs/ecdfs_r/final --n 16 --out samples

The canvas is automatically padded by ``2 * loss_margin`` and the interior kept,
because the score is only correct away from the border.  Samples are shown in
the log representation and in nJy -- the model map is ``exp(x)`` and no band
enters it, so there is nothing per-band to pick.

Every default comes from the checkpoint's own config.  A flag left off is not a
number chosen here; it is the number the model was trained with.
"""

from __future__ import annotations

import argparse

import jax
import numpy as np

from rubin_host_prior.data import LogFluxTransform
from rubin_host_prior.diffusion import VESDE, sample_interior
from rubin_host_prior.training import load_checkpoint


def parser() -> argparse.ArgumentParser:
    """Built separately so a test can ask what a flag defaults to.

    Every flag that names a config field defaults to None and is read from the
    checkpoint's own config instead: the numbers the model was trained with are
    in the checkpoint, and a second set written here would quietly disagree
    with them.  ``--sampler``, ``--weights``, ``--seed`` and ``--out`` are
    choices about this invocation, not about the model.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="samples")
    p.add_argument("--n", type=int, default=None,
                   help="default: the config's train.n_samples")
    p.add_argument("--size", type=int, default=None, help="default: config out_size")
    p.add_argument("--sampler", default="pflow", choices=["pflow", "sde"])
    p.add_argument("--steps", type=int, default=None,
                   help="default: the config's train.sample_steps")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()

    model, config, step = load_checkpoint(args.checkpoint, which=args.weights)
    size = args.size or config.patch.out_size
    n = config.train.n_samples if args.n is None else args.n
    steps = config.train.sample_steps if args.steps is None else args.steps
    print(f"checkpoint step {step}; sampling {n} x {size}x{size} "
          f"(canvas {size + 2 * model.loss_margin}) in {steps} steps")

    x = sample_interior(
        model,
        jax.random.key(args.seed),
        out_size=size,
        n_samples=n,
        sde=VESDE.from_config(config.sde),
        sampler=args.sampler,
        n_steps=steps,
    )
    x = np.asarray(x)[:, 0]
    np.save(f"{args.out}.npy", x)

    transform = LogFluxTransform.from_config(config.transform)
    # x is absolute log flux, so the model map is exp(x) and no band enters it
    # at all -- that is the whole point of the transform.  A band is still worth
    # naming for the sky level it implies, which is what the sample should be
    # read against.
    flux = transform.inverse(x)
    print(f"log-space: mean {x.mean():+.3f} std {x.std():.3f} "
          f"range [{x.min():+.2f}, {x.max():+.2f}]")
    print(f"flux: median {np.median(flux):.2f} nJy, max {flux.max():.1f} nJy")
    print(f"  sky sits at x = {transform.sky_level:.2f}, in every band")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
        for i, ax in enumerate(np.atleast_1d(axes).ravel()):
            ax.set_axis_off()
            if i < len(x):
                ax.imshow(x[i], origin="lower", cmap="magma")
        fig.suptitle(f"prior samples (log space), step {step}")
        fig.tight_layout()
        fig.savefig(f"{args.out}.png", dpi=130)
        print(f"wrote {args.out}.png and {args.out}.npy")
    except ImportError:
        print(f"wrote {args.out}.npy (matplotlib unavailable, no PNG)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Draw scenes from a trained prior and save them as a PNG mosaic plus a .npy.

    python scripts/sample.py --checkpoint runs/ecdfs_r/final --n 16 --out samples

The canvas is automatically padded by ``2 * loss_margin`` and the interior kept,
because the score is only correct away from the border.  Samples are shown in
the log representation and, for the band given by ``--band``, in nJy.
"""

from __future__ import annotations

import argparse

import jax
import numpy as np

from rubin_host_prior.config import BANDS
from rubin_host_prior.data import LogFluxTransform
from rubin_host_prior.diffusion import VESDE, sample_interior
from rubin_host_prior.training import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="samples")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--size", type=int, default=None, help="default: config out_size")
    p.add_argument("--sampler", default="pflow", choices=["pflow", "sde"])
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--band", default="r")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, config, step = load_checkpoint(args.checkpoint, which=args.weights)
    size = args.size or config.patch.out_size
    print(f"checkpoint step {step}; sampling {args.n} x {size}x{size} "
          f"(canvas {size + 2 * model.loss_margin})")

    kwargs = {"n_steps": args.steps}
    x = sample_interior(
        model,
        jax.random.key(args.seed),
        out_size=size,
        n_samples=args.n,
        sde=VESDE.from_config(config.sde),
        sampler=args.sampler,
        **kwargs,
    )
    x = np.asarray(x)[:, 0]
    np.save(f"{args.out}.npy", x)

    transform = LogFluxTransform.from_config(config.transform)
    if args.band not in BANDS:
        raise SystemExit(f"--band must be one of {' '.join(BANDS)}")
    band_idx = np.full(len(x), BANDS.index(args.band))
    # The model itself is band-agnostic; the band only enters when converting
    # back to nJy, through that band's offset.
    flux = transform.inverse(x, band_idx)
    print(f"log-space: mean {x.mean():+.3f} std {x.std():.3f} "
          f"range [{x.min():+.2f}, {x.max():+.2f}]")
    print(f"{args.band}-band flux: median {np.median(flux):.2f} nJy, "
          f"max {flux.max():.1f} nJy")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = int(np.ceil(np.sqrt(args.n)))
        rows = int(np.ceil(args.n / cols))
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

#!/usr/bin/env python
"""Watch real patches go under the forward process, and see what survives where.

    python scripts/forward_diffusion.py --config runs/r/config.json

Rows are patches straight from the loader; columns are the trained schedule's
own noise levels, ascending, each on its own stretch.  Beneath them is the
quantitative version and the one to read first: the signal-to-noise of each
spatial scale against sigma, so you can see the noise level at which structure
of a given size stops being visible at all.

**This is the companion to ``reverse_trajectory.py``.**  Both use
``VESDE.ladder``, so with the same ``--n-sigma`` the columns are the same noise
levels in both figures and can be compared directly.  What a trained model
produces at a given sigma should look like what the forward process leaves
there; the column where the two stop resembling each other is the sigma range
to suspect.

**Point it at a checkpoint, not a config, when there is one.**  ``--checkpoint``
reads ``<checkpoint>/config.json``, which ``save_checkpoint`` writes next to the
weights -- the same file ``reverse_trajectory.py`` loads, so the two figures are
guaranteed to share a schedule, a transform and a patch geometry.  ``--config``
reads whatever file you name, which is right before a run exists and is a way to
disagree with the model once one does.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from rubin_host_prior import plots
from rubin_host_prior.config import Config
from rubin_host_prior.data import LogFluxTransform, PatchDataset, ShardSet
from rubin_host_prior.diffusion import VESDE
from rubin_host_prior.selection import ExtractionConfig

DEFAULT_SHARDS = Path(ExtractionConfig.out) / "shards"


def _schedule_note(config, sde, source) -> str:
    """Exactly what this figure was drawn with, so it can be read against the
    same line printed by ``reverse_trajectory.py``."""
    p = config.patch
    return (
        f"config      {source}\n"
        f"schedule    sigma {sde.sigma_min:.5g} .. {sde.sigma_max:.5g}, "
        f"data mean {sde.data_mean:.4g}\n"
        f"transform   softening {config.transform.softening:.4g} nJy "
        f"(sky at {config.input_offset:.4g} in x)\n"
        f"patches     {p.out_size} px grid, pool {p.pool_factor}, "
        f"native {p.native_size}"
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--checkpoint",
                       help="read the config the model was trained with, from "
                            "beside its weights; the same one "
                            "reverse_trajectory.py uses")
    where.add_argument("--config",
                       help="read a config.json directly, for before a run "
                            "exists")
    p.add_argument("--shards", default=str(DEFAULT_SHARDS),
                   help=f"directory of *.h5 shards (default: {DEFAULT_SHARDS})")
    p.add_argument("--out", default="diagnostics")
    p.add_argument("--n", type=int, default=4, help="patches, one per row")
    p.add_argument("--n-sigma", type=int, default=8, help="columns")
    p.add_argument("--scales", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32],
                   help="spatial scales, in pooled pixels, for the SNR panel")
    p.add_argument("--augment", action="store_true",
                   help="show the patches as training sees them, dihedral and "
                        "translation included; off by default so the figure is "
                        "the same every time")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    source = (Path(args.checkpoint) / "config.json" if args.checkpoint
              else Path(args.config))
    config = Config.load(source)
    sde = VESDE.from_config(config.sde)
    shards = ShardSet.from_dir(args.shards)
    dataset = PatchDataset.from_shards(
        shards, config, LogFluxTransform.from_config(config.transform),
        in_memory=False)

    fig, path = plots.plot_forward_diffusion(
        dataset, sde, n=args.n, n_sigma=args.n_sigma, seed=args.seed,
        scales=tuple(args.scales), augment=args.augment, out=Path(args.out))

    vis = plots.scale_visibility(
        plots.forward_patches(dataset, args.n, args.augment, args.seed),
        tuple(args.scales))
    print(_schedule_note(config, sde, source))
    print(f"{'scale':>7} {'rms signal':>12} {'visible below sigma':>21}  note")
    for scale, sig, s_vis in zip(vis["scales"], vis["signal"],
                                 vis["sigma_visible"]):
        if s_vis <= sde.sigma_min:
            note = "buried at every trained sigma but the very last steps"
        elif s_vis >= sde.sigma_max:
            note = "never buried -- resolved at every trained sigma"
        else:
            # The schedule is log-uniform, so the share of it spent above this
            # crossing is the share of *decades*, not of sigma.
            frac = (math.log(sde.sigma_max / s_vis)
                    / math.log(sde.sigma_max / sde.sigma_min))
            note = (f"buried for the top {100 * frac:.0f}% of the schedule, "
                    f"learnable in the bottom {100 * (1 - frac):.0f}%")
        print(f"{scale:>7} {sig:>12.4g} {s_vis:>21.4g}  {note}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

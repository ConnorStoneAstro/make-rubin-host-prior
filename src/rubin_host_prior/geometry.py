"""Shape arithmetic for a stack of "valid"-mode convolutions summed into an energy.

Everything in this file is a consequence of one fact: with no padding, an energy
that is the sum over a feature map does not weight every input pixel equally.

Notation
--------
``L``      number of shrinking convolution layers
``k``      kernel size (odd, so ``r = (k - 1) // 2`` per layer per side)
``H``      input side length in pixels
``E``      energy-map side length, ``H - 2 * r * L``

Let ``R = r * L`` (the receptive-field radius of one energy-map cell).  Energy-map
cell ``p`` depends on input pixels ``[p, p + 2R]``, so

    dE/dx_i = sum over p in [max(0, i - 2R), min(i, E - 1)]

which contains the full ``2R + 1`` terms only when ``2R <= i <= H - 2R - 1``.

Outside that window the sum runs over a *subset* of the kernel offsets, so a
border pixel's derivative is a different linear functional of the weights than
an interior pixel's.  Note it is not merely smaller: individual kernel weights
have either sign, so a partial sum can exceed the full one in magnitude.  What
matters is that it is systematically *different*, and no amount of training
fixes it -- the network has no path to the missing terms.  Hence:

* the denoising loss is evaluated only on the interior ``H - 4R`` window, and
* to obtain a correct score on a region of interest of size ``Rgn`` you must feed
  the network ``Rgn + 4R`` pixels.
"""

from __future__ import annotations


def layer_radius(kernel_size: int) -> int:
    """Per-side shrinkage of one valid convolution."""
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    return (kernel_size - 1) // 2


def receptive_radius(n_layers: int, kernel_size: int = 3) -> int:
    """``R``: receptive-field radius of a single energy-map cell."""
    return layer_radius(kernel_size) * n_layers


def energy_size(input_size: int, n_layers: int, kernel_size: int = 3) -> int:
    """``E``: side length of the summed energy map."""
    e = input_size - 2 * receptive_radius(n_layers, kernel_size)
    if e < 1:
        raise ValueError(
            f"input_size={input_size} is too small for n_layers={n_layers}, "
            f"kernel_size={kernel_size}; need at least "
            f"{2 * receptive_radius(n_layers, kernel_size) + 1}"
        )
    return e


def min_input_size(n_layers: int, kernel_size: int = 3) -> int:
    """Smallest input that produces any energy at all (a 1x1 energy map)."""
    return 2 * receptive_radius(n_layers, kernel_size) + 1


def loss_margin(n_layers: int, kernel_size: int = 3) -> int:
    """``2R``: pixels to discard on every side before computing the score loss."""
    return 2 * receptive_radius(n_layers, kernel_size)


def interior_size(input_size: int, n_layers: int, kernel_size: int = 3) -> int:
    """``H - 4R``: side length of the fully supported score window."""
    return input_size - 2 * loss_margin(n_layers, kernel_size)


def min_input_for_region(region_size: int, n_layers: int, kernel_size: int = 3) -> int:
    """Input size needed for a correct score over ``region_size`` pixels."""
    return region_size + 2 * loss_margin(n_layers, kernel_size)


def report(
    input_sizes: int | tuple[int, ...],
    n_layers: int,
    kernel_size: int = 3,
) -> str:
    """Multi-line setup summary, printed before training starts.

    The crop is never a free parameter -- it is fixed by ``n_layers`` and
    ``kernel_size`` -- so this is what tells you what changing those did.
    """
    if isinstance(input_sizes, int):
        input_sizes = (input_sizes,)
    r = layer_radius(kernel_size)
    R = receptive_radius(n_layers, kernel_size)
    margin = loss_margin(n_layers, kernel_size)
    lines = [
        "valid-convolution geometry (derived from the architecture, not configured)",
        f"  {n_layers} x {kernel_size}x{kernel_size} valid convolutions"
        f"  ->  receptive radius R = {r}*{n_layers} = {R}",
        f"  loss crop = 2R = {margin} px from every side"
        f"   (beyond 2R the training signal is exactly unbiased)",
        f"  smallest usable patch = 4R + 1 = {4 * R + 1} px",
    ]
    for size in sorted(input_sizes):
        interior = interior_size(size, n_layers, kernel_size)
        if interior <= 0:
            lines.append(
                f"    patch {size}x{size}: TOO SMALL -- needs more than 4R = "
                f"{margin * 2} px per side"
            )
            continue
        e = energy_size(size, n_layers, kernel_size)
        frac = (interior / size) ** 2
        # The score is computed on every pixel and the crop discards the border,
        # so this fraction is also the per-step compute efficiency.
        note = "  <- mostly margin; use larger patches" if frac < 0.10 else ""
        lines.append(
            f"    patch {size:>4}x{size:<4} -> energy map {e}x{e},"
            f" loss on interior {interior}x{interior}"
            f" ({100 * frac:.0f}% of pixels){note}"
        )
    lines.append(
        f"  at inference: a trustworthy region of N px needs a canvas of "
        f"N + {2 * margin} px"
    )
    return "\n".join(lines)

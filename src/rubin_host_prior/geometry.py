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


def describe(input_size: int, n_layers: int, kernel_size: int = 3) -> str:
    """One-line summary, for logs and for sanity-checking a config."""
    r = receptive_radius(n_layers, kernel_size)
    e = energy_size(input_size, n_layers, kernel_size)
    interior = interior_size(input_size, n_layers, kernel_size)
    frac = (interior / input_size) ** 2 if interior > 0 else 0.0
    return (
        f"input {input_size}x{input_size} -> energy map {e}x{e} "
        f"(R={r}, margin={2 * r}); loss interior {interior}x{interior} "
        f"({100 * frac:.0f}% of pixels)"
    )

"""Shape arithmetic for valid-mode convolution stacks summed into an energy.

Everything in this file is a consequence of one fact: with no padding, an energy
that is the sum over a feature map does not weight every input pixel equally.

Notation
--------
``branches``  one dilation tuple per summed stack, e.g. ``((1,)*8, (1,2,4,8,16,1))``
``k``         kernel size (odd, so ``r = (k - 1) // 2`` per layer per side)
``H``         input side length in pixels
``R``         ``r * sum(dilations)`` for a stack; for several, the largest
``E``         energy-map side length, ``H - 2R``

A layer with dilation ``d`` spreads the same ``k`` taps over ``d`` times the
span, so it shrinks a valid map by ``2*r*d`` and contributes ``r*d`` to the
radius.  Reach is therefore the *sum of the dilations*, which is why a doubling
series buys geometric growth for linear depth: ``1,2,4,8,16`` reaches as far as
31 undilated layers.

Energy-map cell ``p`` depends on input pixels ``[p, p + 2R]``, so

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

**Several branches.**  Their energy maps have different sizes, so each is centre
-cropped to the smallest -- the one belonging to the branch with the largest R --
before they are summed.  That leaves the margin at ``2*max(R)``: a short branch's
cells sit centred on the same input pixels as the long branch's, so completeness
for the long branch (``i >= 2R_max``) implies it for the short one
(``i >= R_max + R_short``), and the long branch is always the binding constraint.
"""

from __future__ import annotations

from typing import Sequence

#: One stack's per-layer dilations.
Branches = Sequence[Sequence[int]]


def layer_radius(kernel_size: int) -> int:
    """Per-side shrinkage of one undilated valid convolution."""
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    return (kernel_size - 1) // 2


def branch_radius(dilations: Sequence[int], kernel_size: int = 3) -> int:
    """Receptive radius of one stack: ``r * sum(dilations)``."""
    if not len(dilations):
        raise ValueError("a branch needs at least one layer")
    if any(d < 1 for d in dilations):
        raise ValueError(f"dilations must be >= 1, got {tuple(dilations)}")
    return layer_radius(kernel_size) * int(sum(dilations))


def receptive_radius(branches: Branches, kernel_size: int = 3) -> int:
    """``R``: the largest receptive radius among the summed branches."""
    if not len(branches):
        raise ValueError("an energy needs at least one branch")
    return max(branch_radius(d, kernel_size) for d in branches)


def energy_size(input_size: int, branches: Branches, kernel_size: int = 3) -> int:
    """``E``: side length of the summed energy map."""
    e = input_size - 2 * receptive_radius(branches, kernel_size)
    if e < 1:
        raise ValueError(
            f"input_size={input_size} is too small for this architecture; need "
            f"at least {min_input_size(branches, kernel_size)}"
        )
    return e


def min_input_size(branches: Branches, kernel_size: int = 3) -> int:
    """Smallest input that produces any energy at all (a 1x1 energy map)."""
    return 2 * receptive_radius(branches, kernel_size) + 1


def loss_margin(branches: Branches, kernel_size: int = 3) -> int:
    """``2R``: pixels to discard on every side before computing the score loss."""
    return 2 * receptive_radius(branches, kernel_size)


def interior_size(input_size: int, branches: Branches, kernel_size: int = 3) -> int:
    """``H - 4R``: side length of the fully supported score window."""
    return input_size - 2 * loss_margin(branches, kernel_size)


def min_input_for_region(
    region_size: int, branches: Branches, kernel_size: int = 3
) -> int:
    """Input size needed for a correct score over ``region_size`` pixels."""
    return region_size + 2 * loss_margin(branches, kernel_size)


def report(
    input_sizes: int | tuple[int, ...],
    branches: Branches,
    kernel_size: int = 3,
) -> str:
    """Multi-line setup summary, printed before training starts.

    The crop is never a free parameter -- it is fixed by the dilations and the
    kernel size -- so this is what tells you what changing those did.
    """
    if isinstance(input_sizes, int):
        input_sizes = (input_sizes,)
    r = layer_radius(kernel_size)
    R = receptive_radius(branches, kernel_size)
    margin = loss_margin(branches, kernel_size)
    lines = [
        "valid-convolution geometry (derived from the architecture, not configured)"
    ]
    for i, dil in enumerate(branches):
        rb = branch_radius(dil, kernel_size)
        spread = "x".join(str(d) for d in dil)
        lines.append(
            f"  branch {i}: {len(dil)} x {kernel_size}x{kernel_size}, dilations "
            f"{spread}  ->  R = {r}*{sum(dil)} = {rb}, reach {2 * rb + 1} px"
        )
    if len(branches) > 1:
        lines.append(f"  summed energy: R = max over branches = {R}"
                     f"   (shorter branches are centre-cropped to match)")
    lines += [
        f"  loss crop = 2R = {margin} px from every side"
        f"   (beyond 2R the training signal is exactly unbiased)",
        f"  smallest usable patch = 4R + 1 = {4 * R + 1} px",
    ]
    for size in sorted(input_sizes):
        interior = interior_size(size, branches, kernel_size)
        if interior <= 0:
            lines.append(
                f"    patch {size}x{size}: TOO SMALL -- needs more than 4R = "
                f"{margin * 2} px per side"
            )
            continue
        e = energy_size(size, branches, kernel_size)
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

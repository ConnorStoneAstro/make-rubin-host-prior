"""Reach arithmetic for dilated convolution stacks summed into an energy.

The convolutions are **same-mode, zero-padded**, so shape is no longer the
subject: every layer preserves the grid, the energy map is the size of the
input, and the loss can be taken on every pixel.  What is left to compute is
*reach* -- how far a pixel's score can see -- and how much of that reach lands
on padding rather than on sky.

Notation
--------
``branches``  one dilation tuple per summed stack, e.g. ``((1, 2, 4, 8, 4, 2, 1),)``
``k``         kernel size (odd, so ``r = (k - 1) // 2`` per layer per side)
``H``         grid side length in pixels
``R``         ``r * sum(dilations)`` for a stack; for several, the largest

A layer with dilation ``d`` spreads the same ``k`` taps over ``d`` times the
span, so it contributes ``r*d`` to the radius.  Reach is therefore the *sum of
the dilations*, which is why a doubling series buys geometric growth for linear
depth: ``1,2,4,8,16`` reaches as far as 31 undilated layers, for the same
arithmetic.

**What zero padding costs.**  This module used to open by saying that with no
padding an energy summed over a feature map does not weight every input pixel
equally, and that the fix was to discard ``2R`` pixels from every side of the
loss.  That is still true of valid convolutions, and it is why they were
abandoned: at ``R = 78`` on a 256 px grid the score is within 90% of its plateau
only over a 16x16 window, so there is no patch a 512 px stamp can supply that
would train such a model honestly.

Same-mode padding trades that for a different cost.  Every pixel now gets an
energy, but a pixel within ``R`` of an edge has zeros in its receptive field, so
the energy is no longer translation-equivariant and the network can read its own
distance from the border.  The trade is only sound because the model becomes
**size-locked**: it is trained and evaluated on one grid, with the same padding
in both, so the border is a fixed and known part of the operator rather than a
generalisation gap.  ``padding_fraction`` is how much of the grid is affected.
"""

from __future__ import annotations

from typing import Sequence

#: One stack's per-layer dilations.
Branches = Sequence[Sequence[int]]


def layer_radius(kernel_size: int) -> int:
    """Per-side reach of one undilated convolution."""
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


def real_fraction(input_size: int, radius: int) -> float:
    """Mean fraction of a pixel's receptive field that is data, not padding.

    A pixel at ``p`` sees ``[p - radius, p + radius]``; the part of that outside
    ``[0, H)`` is zeros the padding invented.  Averaged over every pixel of the
    grid and squared for two dimensions, because the window is separable.

    1.0 means no pixel's reach leaves the grid, which for ``radius >= H`` is
    impossible: past ``R = H`` every pixel already sees every other one, so
    further reach buys nothing but more padding.
    """
    if input_size < 1:
        raise ValueError(f"input_size must be positive, got {input_size}")
    span = 2 * radius + 1
    total = sum(
        min(p + radius, input_size - 1) - max(p - radius, 0) + 1
        for p in range(input_size)
    )
    return (total / input_size / span) ** 2


def padding_fraction(
    input_size: int, branches: Branches, kernel_size: int = 3
) -> float:
    """``1 - real_fraction`` at the summed energy's radius: the padding's share."""
    return 1.0 - real_fraction(input_size, receptive_radius(branches, kernel_size))


def report(
    input_sizes: int | tuple[int, ...],
    branches: Branches,
    kernel_size: int = 3,
    loss_margin: int = 0,
) -> str:
    """Multi-line setup summary, printed before training starts.

    Reach follows from the dilations and the kernel size and is never a free
    parameter, so this is what tells you what changing those did.  The loss
    margin *is* free now -- same-mode convolutions give a score everywhere -- so
    it is passed in rather than derived.
    """
    if isinstance(input_sizes, int):
        input_sizes = (input_sizes,)
    r = layer_radius(kernel_size)
    R = receptive_radius(branches, kernel_size)
    lines = ["same-convolution geometry (zero-padded; the grid never changes size)"]
    for i, dil in enumerate(branches):
        rb = branch_radius(dil, kernel_size)
        spread = "x".join(str(d) for d in dil)
        lines.append(
            f"  branch {i}: {len(dil)} x {kernel_size}x{kernel_size}, dilations "
            f"{spread}  ->  R = {r}*{sum(dil)} = {rb}, reach {2 * rb + 1} px"
        )
    if len(branches) > 1:
        lines.append(f"  summed energy: R = max over branches = {R}")
    lines.append(
        f"  score reach = 2R = {2 * R} px"
        f"   (the score is a gradient, so it sees twice the energy's radius)"
    )
    for size in sorted(input_sizes):
        pad = padding_fraction(size, branches, kernel_size)
        note = ""
        if 2 * R >= size:
            note = "  <- reach exceeds the grid; more R buys only padding"
        lines.append(
            f"    grid {size:>4}x{size:<4} -> energy map {size}x{size}, "
            f"{100 * pad:.0f}% of the mean receptive field is padding{note}"
        )
    if loss_margin:
        lines.append(
            f"  loss crop = {loss_margin} px from every side (configured, not "
            f"derived -- every pixel has a score)"
        )
    else:
        lines.append(
            "  loss on every pixel: training and inference use the same grid and "
            "the same padding, so the border is part of the operator, not an "
            "artefact to crop"
        )
    return "\n".join(lines)

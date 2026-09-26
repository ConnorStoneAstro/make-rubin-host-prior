"""Cropping and downsampling from native pixels to the training grid.

Two paths, and the distinction matters for noise fidelity:

``block_mean``
    Exact average pooling by an integer factor.  Independent pixel noise is
    divided by exactly ``factor`` and stays uncorrelated between output pixels.
    This is the default path.

``area_resample``
    Box-filter (area-average) resampling to a non-integer factor, for scale
    jitter.  It is the honest generalisation of average pooling, but output
    pixels whose boundaries fall mid-input-pixel share that pixel, so
    neighbouring outputs become slightly correlated.  That is a real change to
    the noise properties, which is why scale jitter is off by default.

Translations are applied as integer *native*-pixel offsets of the crop.  One
native pixel is 1/3 of an output pixel, so this gives sub-pixel positional
augmentation with no interpolation at all -- strictly free.
"""

from __future__ import annotations

import numpy as np


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Average pool the trailing two axes by an exact integer ``factor``."""
    h, w = a.shape[-2:]
    if h % factor or w % factor:
        raise ValueError(
            f"spatial shape {(h, w)} is not divisible by pool factor {factor}"
        )
    lead = a.shape[:-2]
    return a.reshape(*lead, h // factor, factor, w // factor, factor).mean(
        axis=(-3, -1)
    )


def _area_resample_axis(a: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    """Exact box-filter resample along one axis, via the cumulative sum.

    The integral of the signal over an output pixel's footprint is a difference
    of the cumulative sum evaluated at (generally fractional) edges, so linear
    interpolation of the cumsum gives the exact area average -- no kernel
    approximation, and it works for any real scale factor.  Vectorised over all
    other axes: the interpolation grid is shared, so it is a gather plus a lerp.
    """
    dtype = a.dtype
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    cs = np.concatenate(
        [np.zeros((1,) + a.shape[1:], dtype=np.float64), np.cumsum(a, axis=0)]
    )
    edges = np.linspace(0.0, n, out_size + 1)
    lo = np.minimum(np.floor(edges).astype(np.intp), n)
    hi = np.minimum(lo + 1, n)
    frac = (edges - lo).reshape((out_size + 1,) + (1,) * (cs.ndim - 1))
    interp = cs[lo] * (1.0 - frac) + cs[hi] * frac
    out = np.diff(interp, axis=0) / (n / out_size)
    return np.moveaxis(out, 0, axis).astype(dtype, copy=False)


def area_resample(a: np.ndarray, out_size: int) -> np.ndarray:
    """Box-filter resample the trailing two axes to ``out_size`` square."""
    a = _area_resample_axis(a, out_size, -2)
    return _area_resample_axis(a, out_size, -1)


def crop(a: np.ndarray, size: int, y0: int, x0: int) -> np.ndarray:
    """``size x size`` crop of the trailing two axes with lower corner ``(y0, x0)``."""
    h, w = a.shape[-2:]
    if y0 < 0 or x0 < 0 or y0 + size > h or x0 + size > w:
        raise ValueError(
            f"crop of {size} at ({y0}, {x0}) does not fit in {(h, w)}"
        )
    return a[..., y0 : y0 + size, x0 : x0 + size]


def center_crop(a: np.ndarray, size: int) -> np.ndarray:
    h, w = a.shape[-2:]
    return crop(a, size, (h - size) // 2, (w - size) // 2)


def pool_to_training_grid(
    native: np.ndarray,
    out_size: int,
    pool_factor: int,
    rng: np.random.Generator | None = None,
    translate: bool = False,
    scale_jitter: float = 0.0,
    max_translate: int | None = None,
    context: int = 0,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Native stamp(s) -> ``out_size + 2 * context`` training image(s).

    With ``scale_jitter == 0`` this is a crop plus exact average pooling.  With
    jitter the crop size becomes ``round(out_size * pool_factor * (1 + delta))``
    and an area resample follows; ``delta`` is clipped to whatever the native
    stamp can actually supply, so a too-small ``native_size`` quietly reduces the
    jitter range rather than erroring.

    ``max_translate`` caps how far the crop may wander from the stamp centre,
    in native pixels per side.  Without it a small crop would roam over the whole
    stamp; with it every training size looks at the same neighbourhood.

    **Context.**  The loss discards ``2R`` pixels from every side, so a patch of
    ``out_size`` would train on ``out_size - 4R``.  ``context`` (in *output*
    pixels, and ``2R`` is the only value that makes sense) carries that border
    along so the loss lands on exactly the nominal crop instead.

    It is taken from the rest of the stamp wherever there is any, and only the
    shortfall is ``pad_mode``-filled.  So how much of the border is real depends
    on where the crop landed: centred, it is half real; pushed to one edge, that
    side is wholly synthetic and the opposite side wholly real.  Padding cannot
    manufacture clean signal -- the loss pixels whose receptive field holds no
    synthetic pixel are exactly the ones a bare ``out_size = native // pool``
    crop would have given -- what it buys is training on the rest of the nominal
    crop as well, at the cost of a seam somewhere in those pixels' receptive
    fields.

    ``reflect`` and not ``symmetric``: the border pixel is not duplicated, so no
    column is counted twice.  Mirrored sky is valid sky -- the augmentation
    already teaches dihedral invariance -- and the only thing reflection creates
    that nature does not is the correlation across the seam itself.
    """
    h, w = native.shape[-2:]
    if h != w:
        raise ValueError(f"expected a square native stamp, got {(h, w)}")
    nominal = out_size * pool_factor

    if scale_jitter > 0:
        if rng is None:
            raise ValueError("scale_jitter requires an rng")
        lo = int(np.ceil(nominal / (1.0 + scale_jitter)))
        hi = min(h, int(np.floor(nominal * (1.0 + scale_jitter))))
        size = int(rng.integers(lo, hi + 1)) if hi > lo else min(nominal, h)
    else:
        size = nominal
    if size > h:
        raise ValueError(
            f"need {size} native pixels but the stamp is {h}; "
            f"increase native_size or reduce out_size * pool_factor"
        )

    room = h - size
    if max_translate is not None:
        # Cap the offset rather than letting it use the whole stamp.  A small
        # crop leaves a lot of room, and using all of it would make small
        # patches mostly blank sky far from the host -- i.e. a different data
        # distribution at every size, which is not what varying the size is for.
        room = min(room, 2 * max(max_translate, 0))
    if translate and room > 0:
        if rng is None:
            raise ValueError("translate requires an rng")
        y0 = int(rng.integers(0, room + 1))
        x0 = int(rng.integers(0, room + 1))
    else:
        y0 = x0 = room // 2
    centre_shift = (h - size - room) // 2
    y0 += centre_shift
    x0 += centre_shift

    # The border scales with the crop, so jitter changes the angular size of the
    # context in step with the scene rather than leaving it fixed.  Unjittered
    # this is exactly context * pool_factor.
    pad = int(round(context * size / out_size)) if context else 0
    if pad == 0:
        patch = crop(native, size, y0, x0)
    else:
        want = (y0 - pad, y0 + size + pad, x0 - pad, x0 + size + pad)
        ay0, ay1 = max(want[0], 0), min(want[1], h)
        ax0, ax1 = max(want[2], 0), min(want[3], w)
        patch = native[..., ay0:ay1, ax0:ax1]
        widths = [(0, 0)] * (patch.ndim - 2) + [
            (ay0 - want[0], want[1] - ay1),
            (ax0 - want[2], want[3] - ax1),
        ]
        if any(before or after for before, after in widths[-2:]):
            # `reflect` handles a pad wider than the array by reflecting again,
            # so a crop at the very edge of a small stamp still works -- it just
            # ends up with a mostly synthetic border, which the caller can see
            # from `real_context`.
            patch = np.pad(patch, widths, mode=pad_mode)
    if size == nominal:
        return block_mean(patch, pool_factor)
    return area_resample(patch, out_size + 2 * context)

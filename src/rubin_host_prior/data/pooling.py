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

    **The context is real sky or it is an error.**  This used to reflect the
    shortfall, and the trade looked cheap: the loss pixels whose receptive field
    held no synthetic pixel were exactly the ones a bare crop would have given,
    so reflection appeared to buy the rest of the nominal crop for free.  What
    that reasoning missed is *which* pixels it buys them with.  A reflected
    border is mirror-symmetric about the seam at every scale, and it is the
    large scales that are almost all border -- at R = 36 on a 512 px stamp only
    4% of a 128 px loss region had a receptive field free of it.  So the coarse
    part of the score was fit almost entirely to a symmetry nature does not
    have, which is the leading explanation for three architectures in a row
    whose samples had no structure above ~16 px.

    Reach is therefore bounded by the stamp: ``out_size + 4R`` pooled pixels
    must come out of ``native_size``, and ``max_translate`` has to keep the
    whole window inside it.  ``Config.usable_size_range`` and
    ``Config.max_translate_native`` compute both; this is where it is enforced.
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
        side = size + 2 * pad
        if y0 < pad or x0 < pad or y0 + size + pad > h or x0 + size + pad > w:
            raise ValueError(
                f"a {out_size} px crop with {context} px of context needs "
                f"{side} native px on a side and has to sit inside the "
                f"{h} px stamp, but the crop landed at ({y0}, {x0}). Either the "
                f"stamp is too small for out_size + 2 * context "
                f"({out_size} + 2 * {context} = {out_size + 2 * context} pooled "
                f"px, {(out_size + 2 * context) * pool_factor} native), or "
                f"max_translate let it wander off the edge -- pass "
                f"Config.max_translate_native(), which is sized for exactly "
                f"this. The shortfall is not reflected: the border is real sky "
                f"or there is no batch."
            )
        patch = native[..., y0 - pad : y0 + size + pad, x0 - pad : x0 + size + pad]
    if size == nominal:
        return block_mean(patch, pool_factor)
    return area_resample(patch, out_size + 2 * context)

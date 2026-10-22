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


def diagonal_offset(half: int, min_translate: int) -> int:
    """The smallest ``d >= 0`` with ``hypot(d, d) >= min_translate``, capped at ``half``.

    The deterministic stand-in for a draw from the annulus: the point of the
    allowed region closest to the stamp centre, on the diagonal.  At
    ``min_translate == 0`` it is zero, which is the centred crop the loader has
    always produced when translation is off.
    """
    d = int(np.ceil(max(min_translate, 0) / np.sqrt(2.0)))
    return min(d, max(half, 0))


def _annulus_offset(
    rng: np.random.Generator, half: int, min_translate: int
) -> tuple[int, int]:
    """A uniform integer ``(dy, dx)`` in the box of half-width ``half``, outside
    the disc of radius ``min_translate``.

    Rejection, because the region is a square minus a disc and there is no
    closed form for the uniform integer draw over it.  ``PatchConfig`` keeps
    ``min_shift_native <= max_shift_native``, so the disc fits inside the box's
    inscribed circle and at least ``1 - pi/4 = 21%`` of draws land: 64 tries
    fail with probability ~1e-7, and the fallback is still inside the region.
    """
    r2 = min_translate * min_translate
    for _ in range(64):
        dy = int(rng.integers(-half, half + 1))
        dx = int(rng.integers(-half, half + 1))
        if dy * dy + dx * dx >= r2:
            return dy, dx
    d = diagonal_offset(half, min_translate)
    return d, d


def pool_to_training_grid(
    native: np.ndarray,
    out_size: int,
    pool_factor: int,
    rng: np.random.Generator | None = None,
    translate: bool = False,
    scale_jitter: float = 0.0,
    max_translate: int | None = None,
    min_translate: int = 0,
) -> np.ndarray:
    """Native stamp(s) -> ``out_size`` training image(s).

    With ``scale_jitter == 0`` this is a crop plus exact average pooling.  With
    jitter the crop size becomes ``round(out_size * pool_factor * (1 + delta))``
    and an area resample follows; ``delta`` is clipped to whatever the native
    stamp can actually supply, so a too-small ``native_size`` quietly reduces the
    jitter range rather than erroring.

    ``max_translate`` caps how far the crop may wander from the stamp centre,
    in native pixels per side.  Without it a small crop would roam over the whole
    stamp; with it every training size looks at the same neighbourhood.

    ``min_translate`` is a *floor* on the same displacement, measured as a
    radius: the crop centre is drawn from the square of half-width
    ``max_translate`` with the disc of radius ``min_translate`` punched out.
    Each bound takes the shape its meaning asks for -- the maximum is per axis,
    because that is how a crop runs out of stamp, and the minimum is radial,
    because a square hole would print the pixel axes into the data.  At
    ``min_translate == 0`` the draw is the plain uniform square it has always
    been, byte for byte.

    This is what builds a training set whose galaxies are deliberately
    *off-centre* -- a prior over scenes framed the way a transient is framed,
    some way out from the nucleus but still on the host.  With ``translate``
    off the crop sits at the allowed position nearest the centre rather than at
    the centre itself, so an un-augmented batch is still drawn from the
    distribution the model was trained on.

    ``min_translate`` is clipped to the room the stamp has; ``PatchConfig``
    checks the configured value against the reference size and says so there,
    where the number can be fixed.

    **There is no context border.**  There used to be: valid convolutions gave
    a correct score only ``2R`` pixels inside the input, so the loader carried
    ``2R`` of extra sky on every side for the loss to crop away again.  Where
    the stamp could not supply it the shortfall was reflected, and that put a
    mirror symmetry into exactly the large scales the model was failing to
    learn.  Same-mode convolutions score every pixel, so the crop is now just a
    crop: ``out_size * pool_factor`` native pixels, pooled, and nothing else.
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
    half = room // 2
    min_translate = min(max(int(min_translate), 0), half)
    if translate and room > 0:
        if rng is None:
            raise ValueError("translate requires an rng")
        if min_translate > 0:
            dy, dx = _annulus_offset(rng, half, min_translate)
            y0, x0 = half + dy, half + dx
        else:
            y0 = int(rng.integers(0, room + 1))
            x0 = int(rng.integers(0, room + 1))
    else:
        d = diagonal_offset(half, min_translate)
        y0 = x0 = room // 2 + d
    centre_shift = (h - size - room) // 2
    y0 += centre_shift
    x0 += centre_shift

    patch = crop(native, size, y0, x0)
    if size == nominal:
        return block_mean(patch, pool_factor)
    return area_resample(patch, out_size)

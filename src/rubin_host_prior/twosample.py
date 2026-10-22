"""Does the trained prior actually sample the training distribution?

Every other diagnostic in this package asks a narrow question -- is the sky
scatter right, is the schedule wide enough, is the correlation length inside
the score's reach.  This asks the whole one, and it is the only test here that
can fail for a reason nobody anticipated: **are a batch of model draws and a
batch of training patches the same distribution?**

The tool is PTED (https://github.com/ConnorStoneAstro/pted), a permutation test
on the energy distance.  Energy distance is a proper metric between
distributions -- zero if and only if they are equal -- and the permutation
wrapper turns it into a p-value with no distributional assumptions at all,
which matters because a 16384-dimensional galaxy scene satisfies none of the
usual ones.  The null is "same distribution", so a **small p-value means the
model is wrong**, and a p-value uniform on (0, 1) is the pass.

Three things decide whether a run of this means anything.

**The data sample has to be drawn the way training drew it.**  The model learnt
the *augmented* distribution -- dihedral, translation, and whatever shift
bounds the config sets -- so an un-augmented comparison set is a different
distribution and the test will reject for that reason alone.  ``augment=True``
is the default here and changing it is almost always a mistake.

**The samples have to be independent of each other**, which is why the data
indices are drawn without replacement and the model draws come from one key
split per chunk.

**The harness has to be able to pass.**  ``control`` splits the *data* sample
in two and runs the same test on it, which must give a p-value consistent with
uniform.  If the control fails, nothing about the model has been measured --
something in the comparison itself is broken, and that has to be fixed before
the model's own p-value means anything.  It is cheap next to drawing from the
prior, so it runs by default.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np


def _pted():
    """Imported lazily, with an error that says where to get it."""
    try:
        from pted import pted
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise ImportError(
            "the two-sample test needs PTED >= 2.0: pip install 'pted>=2.0', "
            "or pip install git+https://github.com/ConnorStoneAstro/pted. It "
            "is an optional dependency -- nothing else in this package uses "
            "it. 2.0 is what supplies the JAX backend the exact test needs."
        ) from exc
    return pted


def data_sample(
    dataset, n: int, seed: int = 0, augment: bool = True, chunk: int = 1024,
) -> "jax.Array":
    """``n`` patches from the loader, through the augmentation training uses.

    **The stored patches are not the training distribution.**  Each one stands
    for eight dihedral images times a continuum of crop offsets times its own
    floor draw, and the model learnt that whole thing -- so the loader is an
    effectively infinite generator and ``n`` has no business being capped at
    the shard count.  Indices are drawn uniformly **with replacement**; a base
    patch drawn twice comes back as two different images, which is the point.

    Not ``PatchDataset.batches``, though that is the stream the trainer reads,
    and the difference is worth stating because it is not obvious.  On an
    uncached set ``batches`` draws from a *pool* of ``pool_shards`` resident
    shards, refreshed every ``pool_refill_every`` batches.  Training can
    accept that: over 200k steps the residency correlation averages out.  A
    one-shot draw does not average out -- at the defaults (8 resident,
    refilling every 64 batches) a 16384-patch draw at ``batch_size=512`` is
    32 batches, so it never refills at all and the whole sample comes from 8
    shards of ~31, a quarter of the set.  Shards are written in extraction
    order, so that quarter is a contiguous slice of the sky walk rather than
    a random one.  **The control cannot catch this**: both halves of one
    sample carry the same bias.  Uniform indices cost scattered reads and buy
    coverage, which is the right trade for a diagnostic and the wrong one for
    a training loop.

    ``augment=False`` is the exception.  Without augmentation a base patch
    drawn twice is the *same image twice*, and a permutation test reads exact
    duplicates as agreement -- zero distances in the within-group term, which
    biases the energy distance down. So that path draws without replacement
    and is capped at the stored count, which is honest about what it has.
    """
    rng = np.random.default_rng(seed)
    if augment:
        idx = rng.integers(0, len(dataset), size=n)
    else:
        n = min(n, len(dataset))
        idx = rng.choice(len(dataset), size=n, replace=False)
    out = []
    for start in range(0, n, chunk):
        # Sorted within the chunk for read locality, which costs nothing: the
        # pooled sample is shuffled by the permutation test anyway.
        out.append(dataset.make_batch(np.sort(idx[start:start + chunk]),
                                      rng=rng, augment=augment))
    return jnp.asarray(np.concatenate(out))


def model_sample(
    prior, n: int, seed: int = 0, chunk: int = 64, progress: bool = False,
    **sample_kwargs,
) -> np.ndarray:
    """``n`` draws from the prior, in the model's space.

    Chunked, because the sampler holds the whole batch on the device for every
    step of the ODE and ``n`` large enough for a two-sample test is well past
    what fits.  A fresh key per chunk, so the chunking changes the runtime and
    not the draw's independence.
    """
    key = jax.random.key(seed)
    out = []
    done = 0
    t0 = time.time()
    while done < n:
        take = min(chunk, n - done)
        key, sub = jax.random.split(key)
        out.append(prior.sample(sub, n=take, **sample_kwargs))
        done += take
        if progress:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done}/{n} draws ({rate:.1f}/s)", flush=True)
    return jnp.concatenate(out)


def _as_matrix(a, backend: str):
    """``(N, D)`` in the requested backend, without a round trip it need not make."""
    if backend == "jax":
        # float32: energy distance is a mean over ~1e8 pairwise distances, and
        # PTED's JAX path computes each as ||xi - yj|| directly rather than
        # through the Gram identity, so there is no cancellation to protect
        # against -- the float32 error on a distance is ~sqrt(D) * eps, about
        # 1e-5 relative, which is nine orders below the null's own spread.
        # GPUs are also an order of magnitude slower in float64.
        a = jnp.asarray(a, dtype=jnp.float32)
        return a.reshape(len(a), -1)
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(len(a), -1)


def estimate_cost(n_x: int, n_y: int, dim: int,
                  n_landmarks: int | None = None) -> dict:
    """What the exact test is about to ask for, before it asks.

    Worth printing rather than discovering: the sample is the expensive part
    of this diagnostic, and an out-of-memory failure *after* drawing from the
    prior throws that away.  The distance matrix is ``(n, m)`` floats and the
    work is ``O(n m D)``.
    """
    n = n_x + n_y
    m = n if n_landmarks is None else min(n_landmarks, n)
    return {
        "pooled": n,
        "columns": m,
        "matrix_gb": n * m * 4 / 1e9,
        "samples_gb": n * dim * 4 / 1e9,
        "flops": float(n) * m * dim,
        "exact": n_landmarks is None or n_landmarks >= n,
    }


def compare(
    x,
    y,
    permutations: int = 1000,
    n_landmarks: int | None = None,
    seed: int = 0,
    backend: str = "jax",
) -> dict:
    """PTED between two batches of scenes.  ``x`` is data, ``y`` is the model.

    ``backend="jax"`` keeps both samples as device arrays and lets PTED use its
    own JAX path, so the distance matrix is built wherever the model was
    sampled.  That is the point of the default: the GPU is already set up to
    draw from the prior, and the energy distance is a dense ``O(n m D)``
    reduction, which is exactly what it is good at.  ``backend="numpy"`` runs
    scipy's ``cdist`` in float64 instead -- on a CPU-only machine the two are
    within about 30% of each other, so it is a fallback rather than a penalty.

    ``n_landmarks=None`` is the **exact** test: the full ``(n, n)`` distance
    matrix.  At 16384 data + 512 model that is 1.1 GB of float32 and ~4.7
    Tflop, which a GPU eats and a CPU does not -- measured here, the exact
    test at that size is of order an hour on CPU either backend.  Pass an int
    to fall back to landmarks, which keep the p-value exact (permutations are
    drawn from the subgroup holding the landmark set fixed) and cost
    sensitivity: the null spread grows like ``sqrt(n / m)``, so the detectable
    energy distance goes as ``(n m)^-0.5`` rather than ``n^-1``.
    """
    pted = _pted()
    if backend not in ("jax", "numpy"):
        raise ValueError(f"backend must be 'jax' or 'numpy', got {backend!r}")
    x = _as_matrix(x, backend)
    y = _as_matrix(y, backend)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"the two samples are different scenes: data is {x.shape[1]} "
            f"values per patch and the model gives {y.shape[1]}. The prior is "
            f"size-locked -- check prior.size against patch.out_size."
        )
    pooled = len(x) + len(y)
    if n_landmarks is not None and n_landmarks >= pooled:
        n_landmarks = None  # the exact test, which is what covering it means

    # Centred on the pooled mean.  Energy distance is a function of pairwise
    # distances alone, so one shared scalar subtracted from both samples moves
    # nothing -- and x is absolute log flux sitting near +3 with structure of
    # order 0.4, so taking it off recovers three bits of float32 for free.
    centre = (x.sum() + y.sum()) / float(x.size + y.size)
    x, y = x - centre, y - centre

    t0 = time.time()
    # (statistic, permuted statistics, p-value) -- in that order.  Checked
    # against a known-different pair rather than read off the signature, which
    # says only tuple[float, ndarray, float].
    stat, null, p = pted(x, y, permutations=permutations,
                         n_landmarks=n_landmarks, return_all=True,
                         rng=np.random.default_rng(seed))
    null = np.asarray(null)
    return {
        "p_value": float(p),
        "statistic": float(stat),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        # Where the observed statistic sits in the null, in null sigmas.  Not
        # the test -- the p-value is -- but it says whether a rejection is
        # marginal or absurd, which a p-value floored at 1/(permutations+1)
        # cannot.
        "z": float((stat - null.mean()) / max(null.std(), 1e-300)),
        "n_x": int(len(x)),
        "n_y": int(len(y)),
        "dim": int(x.shape[1]),
        "permutations": int(len(null)),
        "n_landmarks": None if n_landmarks is None else int(n_landmarks),
        "backend": backend,
        "seconds": float(time.time() - t0),
    }


def control(x, permutations: int = 1000,
            n_landmarks: int | None = None, seed: int = 0,
            repeats: int = 5, backend: str = "jax") -> dict:
    """The same test, data against data: the harness's own null.

    Split in half and compared, so by construction the two halves *are* the
    same distribution and the p-value has to look like a draw from U(0, 1).
    A control that rejects means the model's p-value measures nothing until it
    is explained -- correlated rows, a sample drawn two different ways, a
    shape error that survived.

    **Repeated, because one split is not a control.**  At ``alpha = 0.01`` a
    single run rejects by chance one time in a hundred, and a run that does is
    indistinguishable from a broken harness.  Measured here: over fifteen
    splits of one iid sample the z-scores ran from -1.4 to +1.9 with a single
    +3.65, which read alone would have condemned a comparison that was fine.
    Several splits separate the two -- chance rejects one of them, a real
    problem rejects nearly all.

    It needs no draws from the prior, only the data sample already in hand --
    which is why the script runs it *first*.  It exercises the same distance
    matrix at very nearly the same size, so a control that completes is a
    strong signal that the real comparison will fit, and finding out before
    spending the sampling budget is worth the reordering.  It is not free,
    though: ``repeats`` exact tests at this size cost ``repeats`` times what
    the comparison itself does.
    """
    runs = []
    for i in range(max(repeats, 1)):
        rng = np.random.default_rng(seed + 1000 * i)
        order = np.asarray(rng.permutation(len(x)))
        half = len(x) // 2
        runs.append(compare(x[order[:half]], x[order[half:2 * half]],
                            permutations=permutations,
                            n_landmarks=n_landmarks, seed=seed + 1000 * i,
                            backend=backend))
    ps = [r["p_value"] for r in runs]
    return {
        "p_values": ps,
        "z": [r["z"] for r in runs],
        "repeats": len(runs),
        "n_x": runs[0]["n_x"],
        "n_y": runs[0]["n_y"],
        "min_p": float(min(ps)),
        "median_p": float(np.median(ps)),
        "seconds": float(sum(r["seconds"] for r in runs)),
    }


def report(result: dict, control_result: dict | None = None,
           alpha: float = 0.01) -> str:
    """The p-values and what they license you to say."""
    how = ("exact" if result["n_landmarks"] is None
           else f"{result['n_landmarks']} landmarks")
    lines = [
        f"PTED two-sample test: {result['n_x']} data vs {result['n_y']} model "
        f"draws, D = {result['dim']:,}",
        f"  {how}, {result['permutations']} permutations, "
        f"{result['backend']}, {result['seconds']:.1f}s",
        f"  energy distance {result['statistic']:.6g}, null "
        f"{result['null_mean']:.3g} +/- {result['null_std']:.3g}  "
        f"({result['z']:+.1f} sigma)",
        f"  p = {result['p_value']:.4g}",
    ]
    if control_result is not None:
        c = control_result
        ps = " ".join(f"{v:.3g}" for v in c["p_values"])
        lines.append(
            f"  control (data vs data, {c['n_x']} vs {c['n_y']}, "
            f"{c['repeats']} splits): p = {ps}")
        n_reject = sum(v < alpha for v in c["p_values"])
        # One rejection in a handful of splits is what alpha buys you; two is
        # a thousand-to-one coincidence and worth stopping for.
        if n_reject >= 2:
            lines.append(
                f"  the CONTROL rejected on {n_reject} of {c['repeats']} "
                f"splits. Two halves of one sample are the same distribution "
                f"by construction, so this is the comparison or the data "
                f"sample -- correlated rows, a handful of outlier patches "
                f"dominating the energy distance -- and not the model. The "
                f"p-value below measures nothing until it is explained.")
            return "\n".join(lines)
        if n_reject == 1:
            lines.append(
                f"  (one control split of {c['repeats']} rejected, which is "
                f"roughly what alpha = {alpha} buys. Worth a second look if "
                f"the model's p-value is also marginal.)")
    if result["p_value"] < alpha:
        lines.append(
            f"  REJECTED at alpha = {alpha}: the model's draws are not the "
            f"training distribution. The test says only that; which direction "
            f"is a question for the figures -- start with band power by scale "
            f"(plots.schedule_headroom) and the sample grid.")
    else:
        lines.append(
            f"  not rejected at alpha = {alpha}. That is not proof of a match: "
            f"with {result['n_y']} model draws the test sees differences down "
            f"to a limited size, and a larger --n-model is how you look "
            f"harder.")
    return "\n".join(lines)

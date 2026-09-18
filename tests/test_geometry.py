"""The valid-convolution shape arithmetic, and a numerical check of the claim
that motivates the loss crop."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior import geometry as g
from rubin_host_prior.config import EnergyConfig
from rubin_host_prior.nn import ConvEnergyNet, score


def test_shrinkage_matches_a_real_forward_pass(tiny_model):
    for size in (16, 25, 40):
        emap = tiny_model.energy_map(jnp.zeros((1, size, size)), jnp.asarray(1.0))
        assert emap.shape[-1] == g.energy_size(size, tiny_model.n_layers)


def test_margin_is_twice_the_receptive_radius():
    for n_layers in (1, 3, 8):
        r = g.receptive_radius(n_layers)
        assert r == n_layers
        assert g.loss_margin(n_layers) == 2 * r
        assert g.interior_size(64, n_layers) == 64 - 4 * r


def test_min_input_size_is_the_smallest_that_works():
    n_layers = 3
    smallest = g.min_input_size(n_layers)
    model = ConvEnergyNet(
        EnergyConfig(channels=(4,) * n_layers, embed_dim=8, n_fourier=4),
        key=jax.random.key(0),
    )
    assert model.energy_map(
        jnp.zeros((1, smallest, smallest)), jnp.asarray(1.0)
    ).shape[-1] == 1
    # One pixel smaller must raise, not silently return a zero-size energy map
    # (which would make the energy 0 and the score identically zero).
    with pytest.raises(ValueError, match="at least"):
        model.energy_map(jnp.zeros((1, smallest - 1, smallest - 1)), jnp.asarray(1.0))
    with pytest.raises(ValueError, match="at least"):
        score(model, jnp.zeros((1, smallest - 1, smallest - 1)), jnp.asarray(1.0))


def test_even_kernel_rejected():
    with pytest.raises(ValueError, match="odd"):
        g.layer_radius(4)


def test_score_on_a_constant_scene_depends_only_on_edge_truncation(tiny_model):
    """The exact statement of the claim the loss crop rests on.

    For a constant input, ``dE/dx_i`` sums the kernel-offset contributions of
    every energy cell that sees pixel ``i``.  Which offsets are present depends
    only on how the window ``[i-2R, i]`` is truncated by the ends of the energy
    map -- that is, only on ``min(i, 2R)`` and ``min(H-1-i, 2R)``.  So the score
    must be *exactly* constant within each such truncation class, and interior
    pixels (untruncated on both sides, in both axes) form one class.

    Stated this way the test cannot be fooled by coincidence.  Asserting instead
    that every border pixel differs from the interior by some margin does not
    hold: a partial sum over a subset of kernel offsets can land arbitrarily
    close to the full sum for a particular weight draw, and does.

    If this fails, the margin in ``geometry`` is wrong and the training loss is
    being computed on pixels the network structurally cannot get right.
    """
    from collections import defaultdict

    size = 24
    r2 = tiny_model.loss_margin  # == 2R
    s = np.asarray(
        score(tiny_model, jnp.full((1, size, size), 0.3), jnp.asarray(1.0))
    )[0]

    def truncation(i):
        return (min(i, r2), min(size - 1 - i, r2))

    groups = defaultdict(list)
    for i in range(size):
        for j in range(size):
            groups[(truncation(i), truncation(j))].append(s[i, j])

    # Exactly constant within each class, to float32 precision.
    for key, values in groups.items():
        spread = np.ptp(values) / max(abs(np.mean(values)), 1e-12)
        assert spread < 1e-5, f"class {key} spread {spread:.2e}"

    # The interior is one class, and it is the whole H - 4R window.
    interior_key = ((r2, r2), (r2, r2))
    assert len(groups[interior_key]) == (size - 2 * r2) ** 2
    interior = s[r2 : size - r2, r2 : size - r2]
    assert np.allclose(interior, interior.flat[0], rtol=1e-5)

    # Truncation genuinely changes the value: the border as a whole is nowhere
    # near the interior, even if individual pixels can coincide.
    border = np.ones((size, size), dtype=bool)
    border[r2 : size - r2, r2 : size - r2] = False
    deviation = np.abs(s[border] - interior.flat[0]) / abs(interior.flat[0])
    assert np.median(deviation) > 0.1, np.median(deviation)


def test_energy_hessian_bandwidth_is_2r(tiny_model):
    """``dE/dx_i`` depends on input pixels only within ``2R`` -- the Hessian of the
    energy is banded, which is the same fact stated as a derivative."""
    size = 21
    r2 = tiny_model.loss_margin  # == 2R
    x = jnp.zeros((1, size, size))

    def score_at_centre(xx):
        return score(tiny_model, xx, jnp.asarray(1.0))[0, size // 2, size // 2]

    grad = np.asarray(jax.grad(score_at_centre)(x))[0]
    yy, xx = np.mgrid[0:size, 0:size]
    far = (np.abs(yy - size // 2) > r2) | (np.abs(xx - size // 2) > r2)
    assert np.allclose(grad[far], 0.0, atol=1e-7)
    assert np.any(np.abs(grad[~far]) > 0)

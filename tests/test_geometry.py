"""Same-mode reach arithmetic, and numerical checks of the two claims it rests
on: the grid never changes size, and the score still reaches exactly ``2R``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior import geometry as g
from rubin_host_prior.config import EnergyConfig
from rubin_host_prior.nn import ConvEnergyNet, score


def test_the_grid_comes_out_the_size_it_went_in(tiny_model):
    """The point of same-mode padding.  Under valid convolutions this was
    ``size - 2R`` and every caller had to know the architecture to predict it."""
    for size in (8, 16, 25, 40):
        emap = tiny_model.energy_map(jnp.zeros((1, size, size)), jnp.asarray(1.0))
        assert emap.shape[-1] == size


def test_a_scene_smaller_than_the_reach_still_runs(tiny_model):
    """Valid convolutions refused anything below ``2R + 1``, because the energy
    map would have been empty and the score identically zero.  Same-mode
    padding has no such floor -- a 1x1 scene is all padding but it is a scene."""
    assert tiny_model.energy_map(
        jnp.zeros((1, 1, 1)), jnp.asarray(1.0)
    ).shape[-1] == 1


def test_reach_is_the_sum_of_the_dilations():
    """The whole reason for dilation: reach grows with the sum, not the depth,
    so a doubling series buys geometric growth for linear depth."""
    assert g.branch_radius((1, 2, 4, 8, 16, 1)) == 32
    assert g.branch_radius((1,) * 32) == 32  # the same reach, 32 layers deep
    assert g.branch_radius((1, 2, 4, 8, 16), kernel_size=5) == 62
    assert g.branch_radius((1, 2, 4, 8, 16, 32, 8, 4, 2, 1)) == 78  # the default


def test_several_branches_take_the_longest_reach():
    """Their maps are all the size of the scene now, so they simply add -- but
    the summed energy still reaches as far as its longest branch."""
    fine, coarse = (1,) * 8, (1, 2, 4, 8, 16, 1)
    assert g.receptive_radius((fine,)) == 8
    assert g.receptive_radius((coarse,)) == 32
    assert g.receptive_radius((fine, coarse)) == 32


def test_even_kernel_rejected():
    with pytest.raises(ValueError, match="odd"):
        g.layer_radius(4)


def test_padding_fraction_counts_what_the_convolutions_invent():
    """A pixel sees ``[p-R, p+R]``; whatever of that is off the grid is zeros."""
    # R = 0 reaches nothing off-grid, whatever the size.
    assert g.real_fraction(10, 0) == pytest.approx(1.0)
    # R = 1 on a 3 px grid: the middle pixel is whole, the two ends lose a third
    # each, so the 1-D mean is (2/3 + 1 + 2/3)/3 and the 2-D value is its square.
    assert g.real_fraction(3, 1) == pytest.approx(((2 / 3 + 1 + 2 / 3) / 3) ** 2)
    # It falls as the reach grows against a fixed grid, and never reaches 1.
    fracs = [g.real_fraction(64, r) for r in (4, 16, 64)]
    assert fracs[0] > fracs[1] > fracs[2]
    assert all(f < 1.0 for f in fracs)
    # The default architecture on the default grid: a bit over half is padding.
    default = (1, 2, 4, 8, 16, 32, 8, 4, 2, 1)
    assert g.padding_fraction(128, (default,)) == pytest.approx(0.52, abs=0.01)


def test_the_score_still_reaches_exactly_2r(tiny_model):
    """Padding changes what is at the edges, not how far a pixel can see.

    ``dE/dx_i`` depends on input pixels only within ``2R`` -- the Hessian of the
    energy is banded -- and that is a statement about the kernels, so same-mode
    padding leaves it exactly as valid convolutions had it.
    """
    size = 21
    r2 = 2 * tiny_model.receptive_radius
    x = jnp.zeros((1, size, size))

    def score_at_centre(xx):
        return score(tiny_model, xx, jnp.asarray(1.0))[0, size // 2, size // 2]

    grad = np.asarray(jax.grad(score_at_centre)(x))[0]
    yy, xx = np.mgrid[0:size, 0:size]
    far = (np.abs(yy - size // 2) > r2) | (np.abs(xx - size // 2) > r2)
    assert np.allclose(grad[far], 0.0, atol=1e-7)
    assert np.any(np.abs(grad[~far]) > 0)


def test_on_a_constant_scene_the_padding_free_interior_is_uniform(tiny_model):
    """What is left of the old loss-crop claim, and why the margin is now free.

    For a constant input the score at pixel ``i`` is decided by which taps land
    in the padding, so pixels whose whole ``2R`` reach is on the grid must all
    agree exactly, and pixels near the border must not.  Under valid
    convolutions that was the argument for cropping ``2R``: outside the interior
    a pixel's score was a *different linear functional of the weights* that no
    training could fix.  Here the border pixels are merely different, not
    unreachable -- the same grid and the same padding at training and at
    inference -- which is why the loss is taken on all of them.
    """
    size = 24
    r2 = 2 * tiny_model.receptive_radius
    s = np.asarray(
        score(tiny_model, jnp.full((1, size, size), 0.3), jnp.asarray(1.0))
    )[0]

    interior = s[r2 : size - r2, r2 : size - r2]
    assert interior.size > 0
    assert np.allclose(interior, interior.flat[0], rtol=1e-5)

    border = np.ones((size, size), dtype=bool)
    border[r2 : size - r2, r2 : size - r2] = False
    deviation = np.abs(s[border] - interior.flat[0]) / abs(interior.flat[0])
    assert np.median(deviation) > 0.1, np.median(deviation)


def test_report_states_the_reach_and_what_it_costs():
    text = g.report((48, 64, 96), ((1,) * 8,), kernel_size=3)
    assert "same-convolution geometry" in text
    assert "score reach = 2R = 16 px" in text
    assert "energy map 48x48" in text and "energy map 96x96" in text
    # A smaller grid wastes more of its reach on padding than a larger one.
    small = float(text.split("grid   48x48")[1].split("%")[0].split()[-1])
    big = float(text.split("grid   96x96")[1].split("%")[0].split()[-1])
    assert small > big
    assert "loss on every pixel" in text
    assert "loss crop" in g.report(64, ((1,) * 8,), loss_margin=4)


def test_report_tracks_the_hyperparameters():
    """Changing layers, dilations or kernel size must change the reported reach."""
    assert "2R = 6 px" in g.report(64, ((1,) * 3,), kernel_size=3)
    assert "2R = 16 px" in g.report(64, ((1,) * 8,), kernel_size=3)
    assert "2R = 32 px" in g.report(64, ((1,) * 8,), kernel_size=5)
    assert "2R = 64 px" in g.report(300, ((1, 2, 4, 8, 16, 1),))
    # And a second branch is named, so a run's log says what it was given.
    two = g.report(300, ((1,) * 8, (1, 2, 4, 8, 16, 1)))
    assert "branch 0" in two and "branch 1" in two
    assert "dilations 1x2x4x8x16x1" in two


def test_report_flags_a_grid_the_reach_has_outgrown():
    """Past ``2R >= H`` every pixel already sees every other one, so more reach
    is more padding and nothing else -- the default is deliberately here."""
    text = g.report((64, 300), ((1, 2, 4, 8, 16, 1),))   # 2R = 64
    assert "reach exceeds the grid" in text
    assert text.count("reach exceeds the grid") == 1     # not the 300 px grid


def test_a_model_built_from_the_defaults_scores_its_own_grid():
    from rubin_host_prior.config import Config

    c = Config()
    m = ConvEnergyNet(EnergyConfig(
        channels=((4,) * 10,), dilations=c.energy.dilations,
        embed_dim=8, n_fourier=4), key=jax.random.key(0))
    n = c.patch.out_size
    s = score(m, jnp.zeros((1, n, n)), jnp.asarray(1.0))
    assert s.shape == (1, n, n)

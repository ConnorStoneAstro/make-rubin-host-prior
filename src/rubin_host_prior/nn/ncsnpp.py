"""NCSN++ in JAX: a U-Net that is known to encode galaxies.

A port of ``score_models/architectures/ncsnpp.py`` (Adam, ``dev`` branch), which
is itself a port of Song et al.'s ``score_sde_pytorch``.  It is **not** an energy
model: it predicts ``sigma * score`` directly, so the score it defines is not the
gradient of anything and its Jacobian is not symmetric.  That is the trade being
made -- see ``nn.score`` for what is given up and what is bought.

The port is faithful in everything that touches the numbers and simplified
everywhere else.  Kept, because each is load-bearing:

**The epsilon parameterisation.**  The network outputs ``sigma * score``, an
O(1) quantity at every noise level, and the division by ``sigma`` is analytic.
Nothing in the weights has to span the four decades between ``sigma_min`` and
``sigma_max``.  (The energy model already has the equivalent, through
``sigma_scaling = "inverse_sigma"``; this is the one people expect to be the
difference and it is not.)

**GroupNorm at the head of every residual block.**  Nothing normalises the
*input* of the energy model, and with variance-exploding noise the input at
``sigma_max`` is ``sigma_max`` over the data's own spread times the amplitude of
the input at ``sigma_min`` -- about 20x on these stamps.  Measured on the
11 x 64 dilated energy model with the sky sitting at its ``input_offset``, the
activation leaving its last layer runs 0.66 to 50.4 across the schedule, a
**factor of 77**: the network meets a different nonlinear regime at each noise
level and a per-channel FiLM affine is all it has to undo it.  EDM's input
preconditioner, ``c_in = 1 / sqrt(sigma^2 + sigma_data^2)``, flattens that to
1.0x and would be about ten lines in ``ConvEnergyNet.energy_map`` -- scaling the
input is a differentiable reparametrisation, so ``-grad_x E`` stays exact.
Normalising per group at the start of each block removes the scale before any
learned weight sees it -- the input preconditioner (EDM's ``c_in``) arrived at
by a different route.

**A near-linear embedding of the noise level.**  ``fourier_scale = 0.02`` with
``t`` in ``[0, 1]`` makes ``2*pi*w*t`` a fraction of a radian across the whole
schedule, so the sine and cosine features are almost linear in ``t`` and the
score varies smoothly with sigma.  A large scale makes neighbouring noise levels
nearly orthogonal in the embedding, which trains something closer to a lookup
table than a function (Lu & Song 2024, arXiv:2410.11081).

**Progressive input and output skips.**  The scene is FIR-downsampled into a
pyramid and injected at every resolution, and the output is *built* as a
pyramid: the coarsest level emits a 1-channel map first, and each finer level
upsamples it and adds its own correction.  Large-scale structure therefore has a
short, direct path to the output instead of having to survive every
fine-resolution convolution on the way.  This is the part of the architecture
that most directly addresses a model whose samples have no large scales.

**FIR (1, 3, 3, 1) resampling.**  Anti-aliased up- and downsampling.  Plain
strided or nearest-neighbour resampling folds high-frequency power into the
coarse levels, which is exactly the kind of error that shows up as a wrong
spectrum rather than as a wrong-looking image.

**``init_scale = 1e-2`` on the last convolution of every block** and on every
pyramid output, so the network starts near the identity on its skip path.

Simplified away, none of it active in the reference defaults:

* 2D only, 1 input channel (``dimensions``, ``channels`` are gone);
* ``resblock_type`` is always ``"biggan"`` (the default; ``"ddpm"`` used a
  separate resample module);
* ``progressive`` is ``"output_skip"`` or ``"none"``, ``progressive_input`` is
  ``"input_skip"`` or ``"none"``; the ``"residual"`` variants are not ported;
* no conditioning branch beyond the noise level;
* **no dropout.**  It is 0.0 in the reference defaults, and threading a key
  through every block for it would put a key in ``score(model, x, sigma)``,
  which is the one signature the rest of the package is built on.  Worth
  reinstating if this starts overfitting 32k stamps.

**The port is checked against the original, not just against itself.**
``scripts/crosscheck_ncsnpp.py`` copies the reference's weights in module for
module and compares outputs; last run, seven configurations at three noise
levels each agreed to 1.6e-6 relative -- float32 round-off.  Run it after
changing anything structural here, because the mapping it uses is the
reference's module *order* and it fails loudly when the two drift apart.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from ..config import NCSNppConfig
from .energy import EnergyModel
from .layers import FourierFeatures, get_activation

SQRT2 = np.sqrt(2.0)


# -- initialisation --------------------------------------------------------


def _default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], scale: float = 1.0
) -> Array:
    """DDPM's ``variance_scaling(scale, "fan_avg", "uniform")``.

    ``fan_avg``, not the ``fan_in`` this package uses elsewhere: it is what the
    reference implementation uses everywhere, and with ``scale = 1e-2`` on the
    output convolutions it is also what makes a block start near its skip.
    Shapes are ``(out, in, *kernel)`` for convolutions and ``(out, in)`` for
    linear layers, so axis 1 is fan-in and axis 0 fan-out in both cases.
    """
    scale = 1e-10 if scale == 0 else scale
    receptive = int(np.prod(shape)) / shape[1] / shape[0]
    fan_in, fan_out = shape[1] * receptive, shape[0] * receptive
    limit = np.sqrt(3.0 * scale / ((fan_in + fan_out) / 2.0))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _conv(
    in_ch: int, out_ch: int, kernel: int, init_scale: float = 1.0,
    *, use_bias: bool = True, key: PRNGKeyArray,
) -> eqx.nn.Conv2d:
    """Zero-padded same-mode convolution, DDPM init, zero bias."""
    conv = eqx.nn.Conv2d(in_ch, out_ch, kernel, padding="SAME",
                         use_bias=use_bias, key=key)
    conv = eqx.tree_at(lambda m: m.weight, conv,
                       _default_init(key, conv.weight.shape, init_scale))
    if use_bias:
        conv = eqx.tree_at(lambda m: m.bias, conv, jnp.zeros_like(conv.bias))
    return conv


def _linear(in_f: int, out_f: int, *, key: PRNGKeyArray) -> eqx.nn.Linear:
    lin = eqx.nn.Linear(in_f, out_f, key=key)
    lin = eqx.tree_at(lambda m: m.weight, lin,
                      _default_init(key, lin.weight.shape))
    return eqx.tree_at(lambda m: m.bias, lin, jnp.zeros_like(lin.bias))


def _group_norm(channels: int) -> eqx.nn.GroupNorm:
    """The reference's group count: ``max(min(C // 4, 32), 1)``."""
    return eqx.nn.GroupNorm(max(min(channels // 4, 32), 1), channels, eps=1e-6)


# -- FIR resampling --------------------------------------------------------


def _fir_kernel(taps: tuple[int, ...], gain: float = 1.0) -> Array:
    """Separable 1-D taps to a normalised 2-D kernel, scaled by ``gain``."""
    k = np.outer(np.asarray(taps, float), np.asarray(taps, float))
    return jnp.asarray(k / k.sum() * gain)


def _upfirdn2d(
    x: Float[Array, "c h w"], k: Array, up: int = 1, down: int = 1,
    pad: tuple[int, int] = (0, 0),
) -> Float[Array, "c h2 w2"]:
    """Zero-stuff by ``up``, pad, filter by ``k``, keep every ``down``-th.

    The StyleGAN2 primitive, unbatched and channelwise.  Every resampling in
    this file is one call to it, which is the point: upsampling and downsampling
    then use *the same* filter and cannot drift apart.
    """
    c, h, w = x.shape
    if up > 1:
        stuffed = jnp.zeros((c, h, up, w, up), x.dtype)
        x = stuffed.at[:, :, 0, :, 0].set(x).reshape(c, h * up, w * up)
    p0, p1 = pad
    x = jnp.pad(x, ((0, 0), (p0, p1), (p0, p1)))
    # Convolution, not correlation: the kernel is flipped first.  Symmetric for
    # (1, 3, 3, 1), so this is cosmetic -- but it is what the reference does,
    # and an asymmetric fir_kernel would otherwise silently mirror the filter.
    weight = jnp.broadcast_to(
        jnp.flip(k, (0, 1))[None, None].astype(x.dtype), (c, 1) + k.shape)
    y = jax.lax.conv_general_dilated(
        x[None], weight, window_strides=(1, 1), padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"), feature_group_count=c,
    )[0]
    return y[:, ::down, ::down] if down > 1 else y


def fir_downsample(x, taps=(1, 3, 3, 1), factor: int = 2):
    k = _fir_kernel(taps)
    p = k.shape[0] - factor
    return _upfirdn2d(x, k, down=factor, pad=((p + 1) // 2, p // 2))


def fir_upsample(x, taps=(1, 3, 3, 1), factor: int = 2):
    # gain factor**2 so a constant field keeps its value rather than being
    # divided among the factor**2 cells it now occupies.
    k = _fir_kernel(taps, gain=float(factor) ** 2)
    p = k.shape[0] - factor
    return _upfirdn2d(x, k, up=factor,
                      pad=((p + 1) // 2 + factor - 1, p // 2))


def naive_downsample(x, factor: int = 2):
    c, h, w = x.shape
    return x.reshape(c, h // factor, factor, w // factor, factor).mean((2, 4))


def naive_upsample(x, factor: int = 2):
    return jnp.repeat(jnp.repeat(x, factor, axis=1), factor, axis=2)


# -- blocks ----------------------------------------------------------------


class ResnetBlockBigGANpp(eqx.Module):
    """GroupNorm -> act -> (resample) -> conv -> +time -> norm -> act -> conv, + skip.

    Resampling happens on *both* branches and before the first convolution, so
    the block changes resolution without a separate resample layer and the skip
    stays aligned with it.
    """

    norm0: eqx.nn.GroupNorm
    conv0: eqx.nn.Conv2d
    dense: eqx.nn.Linear
    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    shortcut: eqx.nn.Conv2d | None
    up: bool = eqx.field(static=True)
    down: bool = eqx.field(static=True)
    fir: bool = eqx.field(static=True)
    fir_kernel: tuple[int, ...] = eqx.field(static=True)
    skip_rescale: bool = eqx.field(static=True)
    act: callable = eqx.field(static=True)

    def __init__(
        self, in_ch: int, out_ch: int | None, temb_dim: int, act,
        *, up: bool = False, down: bool = False, fir: bool = True,
        fir_kernel: tuple[int, ...] = (1, 3, 3, 1), skip_rescale: bool = True,
        init_scale: float = 1e-2, key: PRNGKeyArray,
    ):
        out_ch = in_ch if out_ch is None else out_ch
        k0, kd, k1, ks = jax.random.split(key, 4)
        self.norm0 = _group_norm(in_ch)
        self.conv0 = _conv(in_ch, out_ch, 3, key=k0)
        self.dense = _linear(temb_dim, out_ch, key=kd)
        self.norm1 = _group_norm(out_ch)
        # Small: the block starts as (almost) its skip connection.
        self.conv1 = _conv(out_ch, out_ch, 3, init_scale, key=k1)
        self.shortcut = (
            _conv(in_ch, out_ch, 1, key=ks)
            if (in_ch != out_ch or up or down) else None
        )
        self.up, self.down = up, down
        self.fir, self.fir_kernel = fir, tuple(fir_kernel)
        self.skip_rescale = skip_rescale
        self.act = act

    def _resample(self, a):
        if self.up:
            return (fir_upsample(a, self.fir_kernel) if self.fir
                    else naive_upsample(a))
        if self.down:
            return (fir_downsample(a, self.fir_kernel) if self.fir
                    else naive_downsample(a))
        return a

    def __call__(
        self, x: Float[Array, "c h w"], temb: Float[Array, " d"]
    ) -> Float[Array, "c2 h2 w2"]:
        h = self.act(self.norm0(x))
        if self.up or self.down:
            h, x = self._resample(h), self._resample(x)
        h = self.conv0(h)
        # The noise level enters as a per-channel bias, once per block.
        h = h + self.dense(self.act(temb))[:, None, None]
        h = self.conv1(self.act(self.norm1(h)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return (x + h) / SQRT2 if self.skip_rescale else x + h


class SelfAttention(eqx.Module):
    """Single-head self-attention over pixels, with a plain skip.

    Used once, at the coarsest resolution, where the token count is small.  The
    output projection is initialised small so the block begins as its skip.
    """

    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d
    channels: int = eqx.field(static=True)

    def __init__(self, channels: int, init_scale: float = 1e-2,
                 *, key: PRNGKeyArray):
        kq, ko = jax.random.split(key)
        qkv = eqx.nn.Conv2d(channels, 3 * channels, 1, key=kq)
        bound = 1.0 / np.sqrt(channels)
        qkv = eqx.tree_at(
            lambda m: m.weight, qkv,
            jax.random.uniform(kq, qkv.weight.shape, minval=-bound, maxval=bound))
        self.to_qkv = eqx.tree_at(lambda m: m.bias, qkv, jnp.zeros_like(qkv.bias))
        out = eqx.nn.Conv2d(channels, channels, 1, key=ko)
        bound = init_scale / np.sqrt(channels)
        out = eqx.tree_at(
            lambda m: m.weight, out,
            jax.random.uniform(ko, out.weight.shape, minval=-bound, maxval=bound))
        self.to_out = eqx.tree_at(lambda m: m.bias, out, jnp.zeros_like(out.bias))
        self.channels = channels

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        c, h, w = x.shape
        q, k, v = jnp.split(self.to_qkv(x), 3, axis=0)
        q = q.reshape(c, h * w).T          # (hw, c)
        k = k.reshape(c, h * w)            # (c, hw)
        v = v.reshape(c, h * w).T          # (hw, c)
        weights = jax.nn.softmax(q @ k / np.sqrt(c), axis=-1)
        attn = (weights @ v).T.reshape(c, h, w)
        return self.to_out(attn) + x


class Combine(eqx.Module):
    """Fold one level of the input pyramid into the trunk."""

    conv: eqx.nn.Conv2d
    method: str = eqx.field(static=True)

    def __init__(self, in_ch: int, out_ch: int, method: str = "cat",
                 *, key: PRNGKeyArray):
        if method not in ("cat", "sum"):
            raise ValueError(f"combine_method must be 'cat' or 'sum', got {method!r}")
        self.conv = _conv(in_ch, out_ch, 1, key=key)
        self.method = method

    def __call__(self, pyramid, h):
        p = self.conv(pyramid)
        return jnp.concatenate([p, h], axis=0) if self.method == "cat" else p + h


class _DownLevel(eqx.Module):
    blocks: tuple[ResnetBlockBigGANpp, ...]
    down: ResnetBlockBigGANpp | None
    combine: Combine | None


class _UpLevel(eqx.Module):
    blocks: tuple[ResnetBlockBigGANpp, ...]
    pyramid_norm: eqx.nn.GroupNorm | None
    pyramid_conv: eqx.nn.Conv2d | None
    up: ResnetBlockBigGANpp | None


# -- the network -----------------------------------------------------------


class NCSNpp(eqx.Module):
    """``x, sigma -> sigma * score``.  See ``score`` for the division.

    **Size-locked**, like the energy model but for a second reason on top of the
    zero padding: the resolution halves ``len(ch_mult) - 1`` times, so the grid
    must be divisible by ``2 ** (len(ch_mult) - 1)``.  ``Config.check_sizes``
    reports a training size that does not, because the failure otherwise
    surfaces as a shape mismatch in the skip concatenation on the way back up.
    """

    time_fourier: FourierFeatures
    time_dense0: eqx.nn.Linear
    time_dense1: eqx.nn.Linear
    conv_in: eqx.nn.Conv2d
    down_levels: tuple[_DownLevel, ...]
    mid_block1: ResnetBlockBigGANpp
    mid_attn: SelfAttention | None
    mid_block2: ResnetBlockBigGANpp
    up_levels: tuple[_UpLevel, ...]
    out_norm: eqx.nn.GroupNorm | None
    out_conv: eqx.nn.Conv2d | None
    config: NCSNppConfig = eqx.field(static=True)
    #: ``log(sigma_min)`` and ``log(sigma_max)``, which turn a sigma into the
    #: ``t in [0, 1]`` the Fourier embedding is scaled for.  Static, and from
    #: the SDE section of the config, for the same reason as ``input_offset``:
    #: they are pytree metadata, so a deserialised checkpoint must get them from
    #: the config that travelled with it rather than from a fresh skeleton.
    log_sigma_min: float = eqx.field(static=True)
    log_sigma_max: float = eqx.field(static=True)
    input_offset: float = eqx.field(static=True)
    act: callable = eqx.field(static=True)

    def __init__(
        self, config: NCSNppConfig, *, sigma_min: float, sigma_max: float,
        input_offset: float = 0.0, key: PRNGKeyArray,
    ):
        config.check()
        if not 0 < sigma_min < sigma_max:
            raise ValueError(
                f"need 0 < sigma_min < sigma_max, got {sigma_min} and {sigma_max}"
            )
        self.config = config
        self.log_sigma_min = float(np.log(sigma_min))
        self.log_sigma_max = float(np.log(sigma_max))
        self.input_offset = float(input_offset)
        self.act = act = get_activation(config.activation)
        nf, mult = config.nf, tuple(config.ch_mult)
        temb_dim = 4 * nf
        n_levels = len(mult)

        def next_key():
            # Split off one at a time rather than pre-splitting a fixed pool:
            # the module count depends on ch_mult and num_blocks, and a pool
            # sized by guesswork is a config away from running out.
            nonlocal key
            key, sub = jax.random.split(key)
            return sub

        # -- noise-level branch --------------------------------------------
        # The same frozen-basis machinery as the energy model's sigma
        # embedding, and frozen for the same reasons (see nn.layers).  nf // 2
        # frequencies, sin and cos, gives nf features.
        self.time_fourier = FourierFeatures(
            nf // 2, config.fourier_scale, config.fourier_seed)
        self.time_dense0 = _linear(nf, temb_dim, key=next_key())
        self.time_dense1 = _linear(temb_dim, temb_dim, key=next_key())

        def block(in_ch, out_ch, **kw):
            return ResnetBlockBigGANpp(
                in_ch, out_ch, temb_dim, act, fir=config.fir,
                fir_kernel=config.fir_kernel, skip_rescale=config.skip_rescale,
                init_scale=config.init_scale, key=next_key(), **kw)

        # -- downsampling --------------------------------------------------
        self.conv_in = _conv(config.in_channels, nf, 3, key=next_key())
        skip_channels = [nf]
        ch = nf
        down = []
        for level in range(n_levels):
            blocks = []
            for _ in range(config.num_blocks):
                blocks.append(block(ch, nf * mult[level]))
                ch = nf * mult[level]
                skip_channels.append(ch)
            resample = combine = None
            if level != n_levels - 1:
                resample = block(ch, None, down=True)
                if config.progressive_input == "input_skip":
                    combine = Combine(config.in_channels, ch,
                                      config.combine_method, key=next_key())
                    if config.combine_method == "cat":
                        ch *= 2
                skip_channels.append(ch)
            down.append(_DownLevel(tuple(blocks), resample, combine))
        self.down_levels = tuple(down)

        # -- bottleneck ----------------------------------------------------
        self.mid_block1 = block(ch, None)
        self.mid_attn = (SelfAttention(ch, config.init_scale, key=next_key())
                         if config.attention else None)
        self.mid_block2 = block(ch, None)

        # -- upsampling ----------------------------------------------------
        up = []
        for level in reversed(range(n_levels)):
            blocks = []
            # num_blocks + 1: one extra to consume the skip left by the
            # downsample/combine at this level.
            for _ in range(config.num_blocks + 1):
                blocks.append(block(ch + skip_channels.pop(), nf * mult[level]))
                ch = nf * mult[level]
            norm = conv = None
            if config.progressive == "output_skip":
                norm = _group_norm(ch)
                conv = _conv(ch, config.in_channels, 3, config.init_scale,
                             key=next_key())
            resample = block(ch, None, up=True) if level != 0 else None
            up.append(_UpLevel(tuple(blocks), norm, conv, resample))
        assert not skip_channels, skip_channels
        self.up_levels = tuple(up)

        # Only when the output is not already built as a pyramid.  init_scale 1
        # here, not 1e-2: this is the sole path to the output, and starting it
        # near zero starts the whole network at score zero.
        if config.progressive == "output_skip":
            self.out_norm = self.out_conv = None
        else:
            self.out_norm = _group_norm(ch)
            self.out_conv = _conv(ch, config.in_channels, 3, 1.0, key=next_key())

    # -- interface ---------------------------------------------------------

    @property
    def in_channels(self) -> int:
        return self.config.in_channels

    @property
    def loss_margin(self) -> int:
        """Zero.  Every pixel is scored; there is no truncated border to crop."""
        return 0

    def t_of_sigma(self, sigma: Float[Array, ""]) -> Float[Array, ""]:
        """The schedule's own time, ``0`` at ``sigma_min`` and ``1`` at ``sigma_max``.

        Conditioning on this rather than on ``log sigma`` is what makes
        ``fourier_scale`` mean the same thing whatever units the data is in: the
        embedding's smoothness is then a property of the architecture and not of
        how bright the sky happens to be.
        """
        return (jnp.log(sigma) - self.log_sigma_min) / (
            self.log_sigma_max - self.log_sigma_min)

    def __call__(
        self, x: Float[Array, "c h w"], sigma: Float[Array, ""]
    ) -> Float[Array, "c h w"]:
        cfg = self.config
        temb = self.time_fourier(self.t_of_sigma(jnp.asarray(sigma)))
        temb = self.time_dense1(self.act(self.time_dense0(temb)))

        # Centred for the same reason as the energy model: every convolution
        # here is zero-padded too, and an uncentred sky makes the border a cliff.
        x = x - self.input_offset
        pyramid_in = x if cfg.progressive_input == "input_skip" else None
        h = self.conv_in(x)
        skips = [h]
        for level in self.down_levels:
            for blk in level.blocks:
                h = blk(h, temb)
                skips.append(h)
            if level.down is not None:
                h = level.down(h, temb)
                if level.combine is not None:
                    pyramid_in = (fir_downsample(pyramid_in, cfg.fir_kernel)
                                  if cfg.fir else naive_downsample(pyramid_in))
                    h = level.combine(pyramid_in, h)
                skips.append(h)

        h = self.mid_block1(h, temb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        pyramid = None
        for level in self.up_levels:
            for blk in level.blocks:
                h = blk(jnp.concatenate([h, skips.pop()], axis=0), temb)
            if level.pyramid_conv is not None:
                contribution = level.pyramid_conv(
                    self.act(level.pyramid_norm(h)))
                if pyramid is None:
                    pyramid = contribution
                else:
                    pyramid = (fir_upsample(pyramid, cfg.fir_kernel)
                               if cfg.fir else naive_upsample(pyramid))
                    pyramid = pyramid + contribution
            if level.up is not None:
                h = level.up(h, temb)
        assert not skips

        if pyramid is not None:
            return pyramid
        return self.out_conv(self.act(self.out_norm(h)))

    def score(
        self, x: Float[Array, "c h w"], sigma: Float[Array, ""]
    ) -> Float[Array, "c h w"]:
        """``net(x, sigma) / sigma``.

        The network predicts ``sigma * score``, which is O(1) at every noise
        level; the decades live in this division and not in the weights.
        """
        return self(x, sigma) / jnp.asarray(sigma)

    # -- reporting ---------------------------------------------------------

    def describe(self, config) -> str:
        from .score import n_parameters

        cfg = self.config
        mult = tuple(cfg.ch_mult)
        grid = max(config.patch.training_sizes)
        resample = (f"FIR {tuple(cfg.fir_kernel)}" if cfg.fir else "naive")
        return "\n".join([
            f"NCSN++: nf {cfg.nf}, ch_mult {mult} -> widths "
            f"{[cfg.nf * m for m in mult]}, {cfg.num_blocks} blocks/level, "
            f"{n_parameters(self):,} parameters",
            f"  resolutions {[grid // 2 ** i for i in range(len(mult))]} from a "
            f"{grid} grid ({grid * config.patch.pool_factor} native); attention "
            f"{'at the bottleneck' if cfg.attention else 'off'}",
            f"  progressive in/out {cfg.progressive_input}/{cfg.progressive}, "
            f"{resample} resampling, host free to move "
            f"+/-{config.patch.max_translate_native // 2} native px",
            f"  sigma -> t in [0, 1] over [{np.exp(self.log_sigma_min):.4g}, "
            f"{np.exp(self.log_sigma_max):.4g}], fourier_scale "
            f"{cfg.fourier_scale}",
        ])

    def log_header(self, config) -> dict:
        cfg = self.config
        return {
            "architecture": "ncsnpp",
            "nf": cfg.nf,
            "ch_mult": list(cfg.ch_mult),
            "num_blocks": cfg.num_blocks,
            "attention": cfg.attention,
            "progressive": cfg.progressive,
            "progressive_input": cfg.progressive_input,
            "fir": cfg.fir,
            "fourier_scale": cfg.fourier_scale,
            "loss_margin": 0,
        }


def scalar_energy(
    h: Float[Array, "c h w"], sigma: Float[Array, ""], form: str,
    data_std: float,
) -> Float[Array, ""]:
    """Turn an NCSN++ output map into a scalar energy.

    Separated from the module so the two formulas can be checked against
    analytic cases without a network in the way -- which is how the ``"dae"``
    prefactor is pinned (see ``NCSNppEnergy``).
    """
    sigma = jnp.asarray(sigma)
    if form == "sum":
        return jnp.sum(h) / sigma
    if form == "dae":
        return 0.5 * (data_std**2 + sigma**2) / sigma**2 * jnp.sum(h**2)
    raise ValueError(f"energy_form must be 'sum' or 'dae', got {form!r}")


class NCSNppEnergy(EnergyModel):
    """The same U-Net, read as a scalar energy instead of as a score.

    Wraps an ``NCSNpp`` rather than subclassing it, so the backbone is byte for
    byte the one that was validated against the reference and a trained
    ``NCSNpp`` checkpoint's weights live at ``.net`` unchanged.  ``score`` comes
    from ``EnergyModel``: ``-grad_x E``, an exact gradient, which is the whole
    point -- **HMC needs the potential itself** for its accept/reject step, not
    just its gradient, and a score network supplies only the gradient.

    The cost is the one the energy model always paid: the loss gradient is a
    gradient of a gradient, so a step is 2-3x an ``NCSNpp`` step and holds more
    activations.  Expect to halve the batch.

    Two ways to make the scalar, ``NCSNppConfig.energy_form``:

    ``"sum"`` -- ``E = sum(h) / sigma``, the network's output map summed over
    the grid, with the same inverse-sigma scaling ``ConvEnergyNet`` uses so that
    ``grad E~`` is O(1) at every noise level.  The most direct reading of the
    network as an energy density, and the most expressive: ``E`` is an
    unconstrained scalar function.  It is also the parameterisation Salimans &
    Ho (2021) found to optimise *worse* than an unconstrained score model --
    nothing in ``-grad_x E`` contains the feed-forward output, so every bit of
    the score has to come back through the whole network.

    ``"dae"`` -- ``E = c(sigma) * ||h||^2 / 2`` with
    ``c(sigma) = (sigma_data^2 + sigma^2) / sigma^2``.  This is Salimans & Ho's
    denoising-autoencoder energy ``||x - s(x)||^2 / 2`` in disguise: Tweedie
    gives the denoiser ``s = x + sigma * h``, so ``x - s = -sigma * h`` and the
    two differ only by the prefactor.  Its gradient is

        -grad E = -c (grad h)^T h

    whose leading term, where ``grad h ~ -I / sigma``, is ``c h / sigma`` --
    **the feed-forward output itself**.  That is the paper's argument for it:
    the score does not have to be reconstructed through the whole network, so
    optimisation behaves like the unconstrained model, and they report it
    performing near-identically to plain epsilon-prediction while admitting an
    explicit energy.

    **The prefactor is not decoration.**  Without it the required ``h`` blows up
    as ``1/sigma``, which is exactly the dynamic range the epsilon
    parameterisation exists to remove.  ``c = (sigma_data^2 + sigma^2)/sigma^2``
    is fixed by demanding that for Gaussian data the *standard* epsilon-optimal
    network be the exact solution: with ``h = -sigma x / (tau^2 + sigma^2)`` and
    ``grad h = -(I - grad D)/sigma``, ``-grad E = -c x sigma^2/(tau^2+sigma^2)^2``,
    which equals the true score ``-x/(tau^2+sigma^2)`` precisely when
    ``c = (tau^2+sigma^2)/sigma^2``.  ``tests/test_ncsnpp.py`` pins that
    identity to machine precision.

    The practical consequence: **the weights of a trained ``NCSNpp`` are
    already the answer** for Gaussian data, and a well-founded starting point
    for anything else.  Real stamps are not Gaussian, so it is an
    initialisation and not a solution -- but one that starts from a model known
    to produce the right spectrum rather than from noise.

    **Measured, and it does not favour ``"dae"`` from scratch.**  150 Adam steps
    on a fixed batch, random init, identical everything else::

        ncsnpp (score)         1.073 -> 0.791 -> 0.591 -> 0.329 -> 0.096
        ncsnpp_energy / sum    0.970 -> 0.819 -> 0.610 -> 0.505 -> 0.311
        ncsnpp_energy / dae    1.013 -> 0.978 -> 0.661 -> 0.590 -> 0.463
                        (steps     1      12      50      100      150)

    A toy problem and too few steps to call a converged result, but the
    ordering is consistent: making the score a gradient costs something, and
    ``"dae"`` costs more again from a *random* init, where ``h`` starts at 0.08
    and its gradient ``-c (grad h)^T h`` is proportional to it.  Raising
    ``init_scale`` does not rescue it -- at 1.0 the score starts at 9.0 against
    a target near 1, with a parameter-gradient norm of 2e4.

    So: **``"sum"`` is the from-scratch form and ``"dae"`` is the warm-start
    form.**  ``"dae"``'s advantage is conditional on ``h`` already being the
    right size, which is exactly what a trained ``"ncsnpp"`` checkpoint
    provides, and exactly the case the exactness result above covers.

    What ``"dae"`` gives up either way: ``E >= 0``, so the highest-probability
    states are exactly where the network predicts zero noise.  A sensible
    inductive bias, and still a restriction ``"sum"`` does not have.
    """

    net: NCSNpp
    #: ``sigma_data``, the per-pixel spread of ``x`` over the training set.
    #: Used only by ``"dae"``; see the prefactor above.
    data_std: float = eqx.field(static=True)

    def __init__(self, config: NCSNppConfig, *, sigma_min: float,
                 sigma_max: float, data_std: float = 1.0,
                 input_offset: float = 0.0, key: PRNGKeyArray):
        if config.energy_form not in ("sum", "dae"):
            raise ValueError(
                f"energy_form must be 'sum' or 'dae', got {config.energy_form!r}")
        if config.energy_form == "dae" and not data_std > 0:
            raise ValueError(
                f"the 'dae' energy needs a positive data_std (it sets the "
                f"prefactor (sigma_data^2 + sigma^2)/sigma^2), got {data_std}"
            )
        self.net = NCSNpp(config, sigma_min=sigma_min, sigma_max=sigma_max,
                          input_offset=input_offset, key=key)
        self.data_std = float(data_std)

    def __call__(
        self, x: Float[Array, "c h w"], sigma: Float[Array, ""]
    ) -> Float[Array, ""]:
        return scalar_energy(self.net(x, sigma), sigma,
                             self.net.config.energy_form, self.data_std)

    # -- interface ---------------------------------------------------------

    @property
    def in_channels(self) -> int:
        return self.net.in_channels

    @property
    def loss_margin(self) -> int:
        return self.net.loss_margin

    @property
    def config(self) -> NCSNppConfig:
        return self.net.config

    def describe(self, config) -> str:
        form = self.net.config.energy_form
        how = ("E = sum(h) / sigma" if form == "sum"
               else f"E = (sigma_d^2 + sigma^2)/(2 sigma^2) * ||h||^2, "
                    f"sigma_d = {self.data_std:.3g}")
        return "\n".join([
            self.net.describe(config).replace("NCSN++:", "NCSN++ energy:", 1),
            f"  scalar energy by '{form}': {how}",
            "  score is -grad_x E, so a training step is a gradient of a "
            "gradient (2-3x an NCSN++ step; halve the batch)",
        ])

    def log_header(self, config) -> dict:
        return {**self.net.log_header(config),
                "architecture": "ncsnpp_energy",
                "energy_form": self.net.config.energy_form,
                "data_std": self.data_std}

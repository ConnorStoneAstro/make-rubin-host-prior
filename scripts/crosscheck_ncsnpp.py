#!/usr/bin/env python
"""Check the JAX NCSN++ against the PyTorch model it was ported from.

Copies the reference's weights into ours module for module and compares the
outputs.  Not a unit test -- it needs ``torch`` and a checkout of the reference
-- but it is the only check that the *port* is faithful rather than merely
self-consistent, so it is worth keeping runnable::

    git clone --depth 1 -b dev https://github.com/AlexandreAdam/score_models
    python scripts/crosscheck_ncsnpp.py --reference score_models/src

Last run: seven configurations, three noise levels each, worst relative
difference **1.6e-6** -- float32 round-off.

The mapping below is the reference's module *order*, since it keeps everything
in one flat ``nn.ModuleList`` and walks it with a counter.  If this script stops
lining up, the construction order in ``nn/ncsnpp.py`` has drifted from it, which
is exactly what the script exists to catch.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _stub_optional_imports() -> None:
    """The reference package pulls in training-only dependencies at import
    time.  Stub the ones that have nothing to do with the architecture rather
    than making the reader install them."""
    peft = types.ModuleType("peft")
    for name in ("PeftModel", "LoraConfig", "get_peft_model"):
        setattr(peft, name, type(name, (), {}))
    sys.modules.setdefault("peft", peft)


def compare(TorchNCSNpp, *, nf=16, ch_mult=(1, 2, 2), num_blocks=2,
            attention=True, fir=True, progressive="output_skip",
            progressive_input="input_skip", combine_method="cat", size=32):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import torch

    from rubin_host_prior.config import NCSNppConfig
    from rubin_host_prior.nn.ncsnpp import NCSNpp

    sigma_min, sigma_max = 0.01, 20.0
    torch.manual_seed(0)
    ref = TorchNCSNpp(
        channels=1, nf=nf, ch_mult=ch_mult, num_blocks=num_blocks,
        attention=attention, dropout=0.0, resblock_type="biggan",
        progressive=progressive, progressive_input=progressive_input,
        combine_method=combine_method, fir=fir, fir_kernel=(1, 3, 3, 1),
        skip_rescale=True, init_scale=1e-2, fourier_scale=0.02,
        activation_type="swish").eval()

    cfg = NCSNppConfig(nf=nf, ch_mult=ch_mult, num_blocks=num_blocks,
                       attention=attention, fir=fir, progressive=progressive,
                       progressive_input=progressive_input,
                       combine_method=combine_method)
    ours = NCSNpp(cfg, sigma_min=sigma_min, sigma_max=sigma_max,
                  key=jax.random.key(0))

    modules = list(ref.all_modules)
    # Our Fourier basis is a *static* field, so it cannot be assigned into;
    # push ours into the reference instead.  Same direction, same comparison.
    modules[0].W.data = torch.tensor(np.array(ours.time_fourier.freqs),
                                     dtype=torch.float32)
    cursor = 1

    def take():
        nonlocal cursor
        cursor += 1
        return modules[cursor - 1]

    edits: list = []
    values: list = []

    def put(getter, tensor):
        # equinox stores a Conv2d bias as (out, 1, 1) where torch stores (out,).
        want = getter(ours).shape
        v = jnp.asarray(tensor.detach().numpy())
        assert v.size == int(np.prod(want)), (v.shape, want)
        edits.append(getter)
        values.append(v.reshape(want))

    def resblock(get, block):
        for ours_name, ref_name in [
            ("norm0", "GroupNorm_0"), ("norm1", "GroupNorm_1"),
            ("dense", "Dense_0"),
        ]:
            sub = getattr(block, ref_name)
            put(lambda m, g=get, n=ours_name: getattr(g(m), n).weight, sub.weight)
            put(lambda m, g=get, n=ours_name: getattr(g(m), n).bias, sub.bias)
        for ours_name, ref_name in [("conv0", "Conv_0"), ("conv1", "Conv_1"),
                                    ("shortcut", "Conv_2")]:
            sub = getattr(block, ref_name, None)
            if sub is None:
                continue
            put(lambda m, g=get, n=ours_name: getattr(g(m), n).weight,
                sub.conv.weight)
            put(lambda m, g=get, n=ours_name: getattr(g(m), n).bias,
                sub.conv.bias)

    d0, d1 = take(), take()
    put(lambda m: m.time_dense0.weight, d0.weight)
    put(lambda m: m.time_dense0.bias, d0.bias)
    put(lambda m: m.time_dense1.weight, d1.weight)
    put(lambda m: m.time_dense1.bias, d1.bias)
    conv_in = take()
    put(lambda m: m.conv_in.weight, conv_in.conv.weight)
    put(lambda m: m.conv_in.bias, conv_in.conv.bias)

    n_levels = len(ch_mult)
    for level in range(n_levels):
        for b in range(num_blocks):
            resblock(lambda m, l=level, b=b: m.down_levels[l].blocks[b], take())
        if level != n_levels - 1:
            resblock(lambda m, l=level: m.down_levels[l].down, take())
            if progressive_input == "input_skip":
                combine = take()
                put(lambda m, l=level: m.down_levels[l].combine.conv.weight,
                    combine.Conv_0.conv.weight)
                put(lambda m, l=level: m.down_levels[l].combine.conv.bias,
                    combine.Conv_0.conv.bias)

    resblock(lambda m: m.mid_block1, take())
    if attention:
        attn = take()
        put(lambda m: m.mid_attn.to_qkv.weight, attn.to_qkv.weight)
        put(lambda m: m.mid_attn.to_qkv.bias, attn.to_qkv.bias)
        put(lambda m: m.mid_attn.to_out.weight, attn.to_out.weight)
        put(lambda m: m.mid_attn.to_out.bias, attn.to_out.bias)
    resblock(lambda m: m.mid_block2, take())

    for i, level in enumerate(reversed(range(n_levels))):
        for b in range(num_blocks + 1):
            resblock(lambda m, i=i, b=b: m.up_levels[i].blocks[b], take())
        if progressive == "output_skip":
            norm, conv = take(), take()
            put(lambda m, i=i: m.up_levels[i].pyramid_norm.weight, norm.weight)
            put(lambda m, i=i: m.up_levels[i].pyramid_norm.bias, norm.bias)
            put(lambda m, i=i: m.up_levels[i].pyramid_conv.weight,
                conv.conv.weight)
            put(lambda m, i=i: m.up_levels[i].pyramid_conv.bias, conv.conv.bias)
        if level != 0:
            resblock(lambda m, i=i: m.up_levels[i].up, take())

    if progressive != "output_skip":
        norm, conv = take(), take()
        put(lambda m: m.out_norm.weight, norm.weight)
        put(lambda m: m.out_norm.bias, norm.bias)
        put(lambda m: m.out_conv.weight, conv.conv.weight)
        put(lambda m: m.out_conv.bias, conv.conv.bias)

    assert cursor == len(modules), (
        f"walked {cursor} of {len(modules)} reference modules: the two "
        f"construction orders have drifted apart")
    ours = eqx.tree_at(lambda m: [g(m) for g in edits], ours, values)

    x = np.random.default_rng(0).standard_normal((3, 1, size, size)).astype(
        np.float32)
    worst = 0.0
    for sigma in (0.02, 1.0, 19.0):
        t = float((np.log(sigma) - np.log(sigma_min))
                  / (np.log(sigma_max) - np.log(sigma_min)))
        with torch.no_grad():
            want = ref(torch.full((3,), t), torch.tensor(x)).numpy()
        got = np.asarray(jax.vmap(ours, in_axes=(0, None))(
            jnp.asarray(x), jnp.float32(sigma)))
        worst = max(worst, float(np.abs(want - got).max() / np.abs(want).max()))
    return worst


CASES = [
    ("defaults", {}),
    ("no attention", dict(attention=False)),
    ("4 levels, 64 px grid", dict(ch_mult=(1, 2, 2, 2), size=64)),
    ("naive resampling", dict(fir=False)),
    ("plain U-Net (no pyramid)", dict(progressive="none",
                                      progressive_input="none")),
    ("combine by sum", dict(combine_method="sum")),
    ("3 blocks per level", dict(num_blocks=3)),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True,
                   help="path to the reference checkout's src/ directory")
    p.add_argument("--tolerance", type=float, default=1e-5)
    args = p.parse_args()

    sys.path.insert(0, args.reference)
    _stub_optional_imports()
    from score_models.architectures.ncsnpp import NCSNpp as TorchNCSNpp

    worst = 0.0
    for name, kwargs in CASES:
        d = compare(TorchNCSNpp, **kwargs)
        worst = max(worst, d)
        print(f"{name:<28} max relative difference {d:.2e}")
    print(f"\nworst overall: {worst:.2e} (tolerance {args.tolerance:.0e})")
    raise SystemExit(0 if worst < args.tolerance else 1)


if __name__ == "__main__":
    main()

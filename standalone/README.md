# `host_prior` — loading a trained Rubin host-galaxy prior

A copy of just enough of `rubin-host-prior` to **rebuild and use** a trained
model, for dropping into an inference package that should not depend on the
training and extraction stack.

```python
from host_prior import load_prior

prior = load_prior("runs/ncsnpp-energy/final")
print(prior.summary())

x = prior.to_x(flux_njy)        # nJy -> the model's space
u = prior.energy(x)             # scalar per scene: HMC's Hamiltonian
g = prior.score(x)              # -grad_x E: its leapfrog
```

Copy the `host_prior/` directory in and rename it to taste — nothing inside
refers to the package by name.

## Dependencies

`jax`, `equinox`, `numpy`. That is the whole list: no h5py, no optax, no
matplotlib, no astropy, nothing from the LSST stack.

## What it carries

The NCSN++ U-Net in its two forms, `ncsnpp` (predicts the score) and
`ncsnpp_energy` (a scalar energy whose gradient is the score), plus the VE
noise schedule, the log-flux transform and the probability-flow sampler. It
does **not** carry the dilated-convolution `energy` architecture; a checkpoint
trained with that is refused by name rather than loaded wrongly.

## The two things to get right

**`to_flux` is `exp(x)` and is not the inverse of `to_x`.** It is the *model
map* — what a forward model calls — and the round trip gives
`soften(f) = s·softplus(f/s)`, not `f`. That gap is the whole discrepancy
between the data and what the prior can express, and it is why a scene drawn
from the prior is strictly positive. Above ~10 softening scales the two agree
to double precision, which is most of a bright galaxy and none of the sky, so
confusing them is a quiet error. Use `to_flux_exact` to recover a measurement,
negative sky pixels included.

**The prior is size-locked.** The convolutions pad with zeros, so the grid the
model was trained on is part of the operator rather than a window onto it.
`prior.size` is that grid. Nothing here stops you evaluating on another one;
the result simply means nothing.

## Keeping it in step with the trainer

`host_prior/_net.py` is **generated**, byte for byte, from
`src/rubin_host_prior/nn/ncsnpp.py` in the training repo, with its three
internal imports repointed. Do not edit it.

The module field declarations in that file *are* the checkpoint format: their
order and their static-ness decide how equinox lays out the pytree, and a copy
that has drifted from the trainer's produces a shape error at best and a wrong
model at worst. So in the training repo:

```bash
python scripts/export_standalone.py          # regenerate
python scripts/export_standalone.py --check  # fail if stale; what CI runs
```

and `tests/test_standalone.py` there loads the same checkpoint through both the
full package and this copy and compares score, energy and samples. When the
architecture changes, re-export and copy `_net.py` across.

The alternative that removes the duplication permanently is to split the
training repo into a small pure-JAX core (model, schedule, transform) that both
it and your inference package depend on. That is more packaging work up front
and nothing here forecloses it.

"""Minimal training loop for the energy-based score model.

Deliberately small: one jitted step, an EMA copy, JSONL logging, periodic
checkpoints.  Nothing clever, so that a tweak to the loss or the schedule is a
three-line change rather than an archaeology exercise.

One cost worth knowing about: the score is already a gradient of the network, so
the loss gradient is a second derivative.  Every step is a
gradient-of-a-gradient, roughly 2-3x the cost of a conventional score network of
the same size.  That is the price of an exactly conservative score.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .. import geometry
from ..config import Config, TrainConfig
from ..diffusion.loss import dsm_loss, dsm_loss_by_sigma
from ..diffusion.sde import VESDE
from ..nn.energy import ConvEnergyNet, n_parameters
from .checkpoint import save_checkpoint
from .ema import ema_decay_at, ema_update


def make_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
    if cfg.cosine_decay:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=max(cfg.warmup_steps, 1),
            decay_steps=cfg.steps,
            end_value=cfg.learning_rate * 0.05,
        )
    else:
        schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=cfg.learning_rate,
            transition_steps=max(cfg.warmup_steps, 1),
        )
    chain = []
    if cfg.grad_clip > 0:
        chain.append(optax.clip_by_global_norm(cfg.grad_clip))
    chain.append(optax.adamw(schedule, weight_decay=cfg.weight_decay))
    return optax.chain(*chain)


def train(
    model: ConvEnergyNet,
    batches: Iterator[np.ndarray],
    config: Config,
    out_dir: str | Path,
    sde: VESDE | None = None,
    eval_batch: np.ndarray | None = None,
    eval_every: int = 0,
    on_log: Callable[[dict], None] | None = None,
    verbose: bool = True,
) -> tuple[ConvEnergyNet, ConvEnergyNet]:
    """Train and return ``(model, ema_model)``.

    ``batches`` is any iterator of ``(B, C, H, W)`` arrays already in the log-space
    training representation.  ``H`` must exceed ``4R`` or the loss has no interior
    (see ``geometry``); this is checked once, up front, rather than producing a
    confusing error 10 000 steps in.
    """
    sde = sde or VESDE.from_config(config.sde)
    cfg = config.train
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config.json")

    # The crop is a function of the architecture, never a configured number.
    # Print it, so that changing n_layers or kernel_size announces what it did
    # rather than silently changing how much of each patch is trained on.
    margin = model.loss_margin
    sizes = config.patch.training_sizes
    setup = geometry.report(sizes, model.n_layers, model.config.kernel_size)
    if verbose:
        print(setup)
        if len(sizes) > 1:
            print(f"  {len(sizes)} training sizes -> {len(sizes)} jit compilations "
                  f"of the train step, cycled round-robin across batches")
        # Step cost scales with batch x H^2, so this is the number to watch when
        # a bigger patch size starts exhausting device memory.
        biggest = max(sizes)
        print(f"  batch {cfg.batch_size} x {biggest}x{biggest} = "
              f"{cfg.batch_size * biggest ** 2:,} score pixels per step "
              f"(cost scales with this; halve the batch if memory is tight)")
        for warning in config.check_sizes():
            print(f"  WARNING: {warning}")
    _validate_geometry(sizes, model)
    optimizer = make_optimizer(cfg)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    ema_model = model

    @eqx.filter_jit
    def step_fn(model, ema_model, opt_state, batch, key, decay):
        loss, grads = eqx.filter_value_and_grad(dsm_loss)(
            model, batch, key, sde, margin
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        ema_model = ema_update(ema_model, model, decay)
        return model, ema_model, opt_state, loss

    @eqx.filter_jit
    def eval_fn(model, batch, key, sigmas):
        # One fixed sigma per example, spanning the schedule: a loss curve
        # against sigma shows *where* the model is underfit, which the scalar
        # training loss hides entirely.
        return dsm_loss_by_sigma(model, batch, sigmas, key, sde, margin)

    key = jax.random.key(cfg.seed)
    checkpoints = set(cfg.checkpoint_steps())
    if verbose and checkpoints:
        print(f"  {len(checkpoints)} checkpoints at steps "
              f"{sorted(checkpoints)[:3]}...{max(checkpoints)}")
        if cfg.n_samples:
            # Said up front, because it is easy to ask for far more than
            # intended and the first checkpoint is a long way into the run.
            canvas = config.patch.out_size + 2 * margin
            print(f"  each draws {cfg.n_samples} samples on a {canvas}x{canvas} "
                  f"canvas ({config.patch.out_size} + 4R) in "
                  f"{2 * cfg.sample_steps} batched backward passes; "
                  f"--n-samples 0 to skip")
        else:
            print("  no samples (n_samples = 0)")
    log_path = out / "log.jsonl"
    running = None
    t0 = time.time()

    with log_path.open("a") as log_file:
        header = {
            "event": "start",
            "n_parameters": n_parameters(model),
            "n_layers": model.n_layers,
            "kernel_size": model.config.kernel_size,
            "receptive_radius": model.receptive_radius,
            "loss_margin": margin,
            "training_sizes": list(sizes),
            "interior_sizes": [
                geometry.interior_size(s, model.n_layers, model.config.kernel_size)
                for s in sizes
            ],
            "sigma_min": sde.sigma_min,
            "sigma_max": sde.sigma_max,
        }
        log_file.write(json.dumps(header) + "\n")
        log_file.flush()

        for step in range(1, cfg.steps + 1):
            batch = jnp.asarray(next(batches))
            if step == 1:
                _check_batch(batch, model)
            key, k_step = jax.random.split(key)
            decay = ema_decay_at(step - 1, cfg.ema_decay)
            model, ema_model, opt_state, loss = step_fn(
                model, ema_model, opt_state, batch, k_step, decay
            )

            loss = float(loss)
            running = loss if running is None else 0.98 * running + 0.02 * loss
            if step % cfg.log_every == 0 or step == 1:
                record = {
                    "step": step,
                    "loss": loss,
                    "loss_ema": running,
                    "seconds": time.time() - t0,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if on_log is not None:
                    on_log(record)

            if eval_every and eval_batch is not None and step % eval_every == 0:
                key, k_eval = jax.random.split(key)
                n = eval_batch.shape[0]
                sigmas = sde.sigma(jnp.linspace(0.0, 1.0, n))
                per = eval_fn(ema_model, jnp.asarray(eval_batch), k_eval, sigmas)
                record = {
                    "step": step,
                    "event": "eval",
                    "sigma": [float(s) for s in sigmas],
                    "loss_by_sigma": [float(v) for v in per],
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if on_log is not None:
                    on_log(record)

            if step in checkpoints:
                # Kept, named by step, so the run leaves a history rather than
                # one overwritten directory -- but without the optimiser state,
                # which is two more copies of the parameters and is only ever
                # needed for the most recent one.  That lives in `latest`, which
                # is overwritten, so ten checkpoints cost the weights ten times
                # and the optimiser once.
                save_checkpoint(out / "checkpoints" / f"step-{step:08d}",
                                step, config, model, ema_model)
                save_checkpoint(out / "latest", step, config, model, ema_model,
                                opt_state)
                record = {"step": step, "event": "checkpoint",
                          "seconds": time.time() - t0}
                if cfg.n_samples:
                    record.update(_write_samples(
                        ema_model, sde, config, out, step, key, verbose))
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if on_log is not None:
                    on_log(record)

    save_checkpoint(out / "final", cfg.steps, config, model, ema_model, opt_state)
    return model, ema_model


def _write_samples(ema_model, sde, config: Config, out: Path, step: int,
                   key, verbose: bool) -> dict:
    """Draw from the EMA weights and write an ``n x n`` grid of them.

    The EMA, not the live weights: it is what inference uses, so it is what a
    picture of progress should show.

    Never fatal.  A diagnostic that kills a run hours in is worse than no
    diagnostic -- sampling is the one thing here that can exhaust device memory
    on its own, since the canvas is 4R larger than the sample and every score
    evaluation sees all of it at once.  A failure is logged with its reason and
    training continues.
    """
    cfg = config.train
    t0 = time.time()
    try:
        from ..diffusion.sampler import sample_interior

        x = sample_interior(
            ema_model,
            jax.random.fold_in(key, step),
            out_size=config.patch.out_size,
            n_samples=cfg.n_samples,
            sde=sde,
            n_steps=cfg.sample_steps,
        )
        x = np.asarray(x)
        from .. import plots

        _, path = plots.plot_samples(x, step=step, out=out / "samples",
                                     name=f"step-{step:08d}")
        if verbose:
            print(f"  step {step:,}: {cfg.n_samples} samples in "
                  f"{time.time() - t0:.0f}s -> {path}")
        return {
            "samples": str(path),
            "sample_seconds": round(time.time() - t0, 1),
            "sample_mean": float(np.mean(x)),
            "sample_std": float(np.std(x)),
            # A sampler that diverges produces inf or nan rather than an error,
            # and a grid of them looks like a blank figure.  Count them.
            "sample_nonfinite": int(np.sum(~np.isfinite(x))),
        }
    except Exception as exc:
        log = f"sampling failed at step {step}: {exc!r}"
        if verbose:
            print(f"  WARNING: {log}")
        return {"sample_error": repr(exc)}


def _validate_geometry(sizes, model: ConvEnergyNet) -> None:
    """Reject configured sizes that leave no interior, before any training.

    Checked up front rather than on the first batch of each size, so a mixed-size
    run does not fail thousands of steps in when the smallest size first comes
    round.
    """
    need = 2 * model.loss_margin + 1
    bad = [s for s in sizes if s < need]
    if bad:
        raise ValueError(
            f"patch size(s) {bad} leave no interior for the loss: a model with "
            f"{model.n_layers} {model.config.kernel_size}x{model.config.kernel_size} "
            f"layers crops {model.loss_margin} px per side, so it needs more than "
            f"4R = {2 * model.loss_margin} px per side. Use larger patches or "
            f"fewer layers."
        )


def _check_batch(batch: jnp.ndarray, model: ConvEnergyNet) -> None:
    if batch.ndim != 4:
        raise ValueError(f"expected (B, C, H, W) batches, got shape {batch.shape}")
    if batch.shape[1] != model.config.in_channels:
        raise ValueError(
            f"batch has {batch.shape[1]} channels, model expects "
            f"{model.config.in_channels}"
        )
    h, w = batch.shape[-2:]
    need = 2 * model.loss_margin + 1
    if min(h, w) < need:
        raise ValueError(
            f"patches are {h}x{w} but a model with {model.n_layers} layers needs "
            f"at least {need} pixels per side to leave any interior for the loss; "
            f"use larger patches or fewer layers"
        )

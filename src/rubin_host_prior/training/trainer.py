"""Minimal training loop, for whichever architecture computes the score.

Deliberately small: one jitted step, an EMA copy, JSONL logging, periodic
checkpoints.  Nothing clever, so that a tweak to the loss or the schedule is a
three-line change rather than an archaeology exercise.

Nothing here knows which model it is training.  It goes through ``nn.score``,
asks the model to ``describe`` itself for the header, and is otherwise the same
loop for an energy and for a U-Net -- which is what makes the two comparable:
same loss, same schedule, same optimiser, same evaluation.

One cost worth knowing about, and it belongs to one of them: ``ConvEnergyNet``'s
score is already a gradient of the network, so the loss gradient is a second
derivative and every step is a gradient-of-a-gradient, roughly 2-3x the cost of
a conventional score network of the same size.  That is the price of an exactly
conservative score.  ``NCSNpp`` predicts the score and pays none of it.
"""

from __future__ import annotations

import json
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from ..config import Config, TrainConfig
from ..diffusion.loss import (dsm_loss, dsm_loss_by_sigma,
                              gaussian_loss_floor)
from ..diffusion.sde import VESDE
from ..nn.score import ScoreModel, n_parameters
from .checkpoint import load_checkpoint, load_opt_state, save_checkpoint
from .ema import ema_decay_at, ema_update


#: Signals that mean "stop at the next step boundary and checkpoint".  SIGUSR1
#: is what Slurm sends with ``--signal=B:USR1@<seconds>``, ahead of the wall
#: clock, which is the whole point: a scheduler that kills the job outright
#: leaves the last checkpoint however old it was.
STOP_SIGNALS: tuple[int, ...] = (signal.SIGUSR1,)


class _StopFlag:
    """Set by a signal handler; read between steps.

    The handler does nothing but set a bool.  Saving a checkpoint from inside a
    handler would run arbitrary JAX and filesystem work at an arbitrary point in
    a training step -- possibly mid-``block_until_ready``, possibly mid-write --
    so the handler records the request and the loop acts on it where the state
    is consistent.
    """

    def __init__(self) -> None:
        self.requested: int | None = None

    def __bool__(self) -> bool:
        return self.requested is not None

    @property
    def name(self) -> str:
        return signal.Signals(self.requested).name if self.requested else ""


@contextmanager
def _catch_stop(signals: Sequence[int], verbose: bool):
    """Install the stop handlers for the duration of a run, then put back
    whatever was there before -- this is a library function, and leaving a
    process's signal disposition changed behind it is not its business."""
    flag = _StopFlag()

    def handler(sig, _frame):
        flag.requested = sig

    previous = {}
    for sig in signals:
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError) as exc:
            # Not the main thread, or the platform has no such signal.  Not
            # fatal: training simply cannot be asked to stop early.
            if verbose:
                print(f"  WARNING: cannot catch {sig}: {exc!r}")
    try:
        yield flag
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


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


def _shardings(n_devices: int | None, batch_size: int, verbose: bool):
    """Return ``(n, replicated, over_batch)`` for plain data parallelism.

    Replicated parameters, a batch split evenly across devices, gradients
    all-reduced every step.  There is no local-SGD mode and no averaging
    interval: the parameters are 4 MB, so an all-reduce is tens of microseconds
    against a step that is a gradient-of-a-gradient, and the exact gradient is
    free.  Periodic averaging exists to hide a slow interconnect between nodes,
    which is not the situation on one node.

    Almost none of this file knows about it, because nothing else has to.  The
    mean in ``dsm_loss`` over a sharded batch axis is what makes XLA insert the
    all-reduce; the weights are then identical on every device, so the EMA, the
    checkpoints and the sampler are unchanged.  In particular a checkpoint says
    nothing about how many devices wrote it -- a chunk trained on four GPUs
    resumes on one, which is what a scheduler that gives you what it has
    requires.

    ``batch_size`` is the *global* batch, split evenly.  Four devices make the
    same run go faster; they do not quadruple the batch behind the learning
    rate.  Raise ``--batch-size`` deliberately if that is what you want.
    """
    available = jax.local_device_count()
    n = available if n_devices is None else int(n_devices)
    if not 1 <= n <= available:
        kinds = sorted({d.device_kind for d in jax.local_devices()})
        raise ValueError(
            f"asked for {n} devices; JAX can see {available} ({', '.join(kinds)})"
        )
    if batch_size % n:
        raise ValueError(
            f"batch size {batch_size} does not divide across {n} devices. It is "
            f"the global batch, split evenly, so it must be a multiple of the "
            f"device count -- raise it to {batch_size + n - batch_size % n} or "
            f"pass fewer devices."
        )
    # Auto, not the default Explicit: `filter_shard` is a sharding *constraint*,
    # which is only defined on auto axes.
    mesh = jax.make_mesh((n,), ("batch",), axis_types=(AxisType.Auto,),
                         devices=jax.devices()[:n])
    if verbose and n > 1:
        print(f"  {n} devices, {batch_size // n} of the {batch_size} batch each, "
              f"gradients all-reduced every step")
    return (n, NamedSharding(mesh, PartitionSpec()),
            NamedSharding(mesh, PartitionSpec("batch")))


def train(
    model: ScoreModel,
    batches: Iterator[np.ndarray],
    config: Config,
    out_dir: str | Path,
    sde: VESDE | None = None,
    eval_batch: np.ndarray | None = None,
    on_log: Callable[[dict], None] | None = None,
    verbose: bool = True,
    resume: str | Path | None = None,
    stop_signals: Sequence[int] = STOP_SIGNALS,
    n_devices: int | None = None,
) -> tuple[ScoreModel, ScoreModel]:
    """Train and return ``(model, ema_model)``.

    ``batches`` is any iterator of ``(B, C, H, W)`` arrays already in the log-space
    training representation.  ``H`` must exceed ``4R`` or the loss has no interior
    (see ``geometry``); this is checked once, up front, rather than producing a
    confusing error 10 000 steps in.

    **Resuming.**  ``resume`` is a checkpoint directory -- ``<out>/latest`` is the
    one to use, since it is the only one carrying the optimiser state.  Four
    things have to come back or the resumed run is not a continuation of the old
    one:

    * the **weights** and the **EMA copy**, which are two separate sets;
    * the **optimiser state**, which holds Adam's moments *and* the learning-rate
      schedule's step count -- restart it and the warmup runs again from zero and
      the cosine decay restarts, which is a different training run;
    * the **step number**, so the EMA warmup (``ema_decay_at``) continues rather
      than treating a half-trained model as a fresh one;
    * the **RNG**, folded with the step so a resumed run does not replay the same
      noise draws it already used.

    ``config.train.steps`` stays the length of the *whole* run across every
    chunk, not the length of one chunk: it is what the schedule decays over and
    what the checkpoint spacing is computed from.  A chunk runs from where the
    last one stopped until the target, or until a stop signal.

    **Stopping early.**  On any of ``stop_signals`` (default ``SIGUSR1``, which is
    what ``sbatch --signal=B:USR1@300`` sends ahead of the wall clock) the loop
    finishes the step it is on, writes ``latest`` with the optimiser state, and
    returns.  It does *not* write ``final``, because the run is not finished --
    that is what tells the next chunk there is more to do.  No samples are drawn
    on the way out: the point is to be gone inside the grace period.

    **Devices.**  ``n_devices`` defaults to every device JAX can see, which on a
    whole NERSC node is all four GPUs and on a laptop is one.  The parameters
    are replicated and the batch is split, so the run is the same run either
    way -- same global batch, same learning rate, same loss curve, just faster.
    ``n_devices=1`` is the single-device path, bit-identical to not having this
    at all.  It is an argument and not a config field on purpose: how many GPUs
    an allocation happened to contain is a property of the machine, not of the
    run, and writing it into the checkpoint would make a 4-GPU chunk look
    incompatible with a 1-GPU one.

    The evaluation batch and the checkpoint samples are left replicated rather
    than sharded.  They run every few thousand steps, so the redundant work is
    invisible, and not sharding them means neither ``--eval-size`` nor
    ``--n-samples`` acquires a divisibility rule.
    """
    sde = sde or VESDE.from_config(config.sde)
    cfg = config.train
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config.json")

    # Printed, so that changing the architecture, the dilations or the margin
    # announces what it did.  The model describes itself: the two architectures
    # have nothing in common to report, and an isinstance ladder here would grow
    # a branch for every one added.
    margin = model.loss_margin
    sizes = config.patch.training_sizes
    if verbose:
        print(model.describe(config))
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
    n_devices, replicated, over_batch = _shardings(
        n_devices, cfg.batch_size, verbose)
    optimizer = make_optimizer(cfg)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    ema_model = model
    start_step = 0
    if resume is not None:
        model, ema_model, opt_state, start_step = _resume(
            resume, config, model, optimizer, verbose)
        if start_step >= cfg.steps:
            raise ValueError(
                f"{resume} is already at step {start_step} and train.steps is "
                f"{cfg.steps}; there is nothing left to do. Raise steps to "
                f"continue this run, or point --out somewhere new to start over."
            )
    model, ema_model, opt_state = eqx.filter_shard(
        (model, ema_model, opt_state), replicated)

    @eqx.filter_jit
    def step_fn(model, ema_model, opt_state, batch, key, decay):
        batch = eqx.filter_shard(batch, over_batch)
        loss, grads = eqx.filter_value_and_grad(dsm_loss)(
            model, batch, key, sde, margin
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        ema_model = ema_update(ema_model, model, decay)
        # Pin the layout rather than leave XLA to infer it every compilation:
        # everything the loop carries between steps is replicated, and the only
        # thing crossing the interconnect is the gradient all-reduce implied by
        # the mean over the sharded batch axis.
        model, ema_model, opt_state = eqx.filter_shard(
            (model, ema_model, opt_state), replicated)
        return model, ema_model, opt_state, loss

    @eqx.filter_jit
    def eval_fn(model, batch, key, sigmas):
        # Every patch at every sigma, so averaging over the batch leaves the
        # sigma dependence and nothing else.  Pairing one patch with each sigma
        # -- which this did -- produced a curve whose bumps were the difference
        # between a blank patch and a bright one, not a property of the model.
        return dsm_loss_by_sigma(model, batch, sigmas, key, sde, margin)

    # Folded with the step the run starts from, so a resumed chunk does not
    # replay the noise draws the previous one already used.
    key = jax.random.fold_in(jax.random.key(cfg.seed), start_step)
    checkpoints = set(cfg.checkpoint_steps())
    if verbose and checkpoints:
        print(f"  {len(checkpoints)} checkpoints at steps "
              f"{sorted(checkpoints)[:3]}...{max(checkpoints)}")
        if cfg.n_samples:
            # Said up front, because it is easy to ask for far more than
            # intended and the first checkpoint is a long way into the run.
            # The grid is the training grid and cannot be anything else: the
            # model is size-locked by its own padding.
            grid = config.patch.out_size
            print(f"  each draws {cfg.n_samples} samples on the {grid}x{grid} "
                  f"training grid in {2 * cfg.sample_steps} batched score "
                  f"evaluations; --n-samples 0 to skip")
        else:
            print("  no samples (n_samples = 0)")
    log_path = out / "log.jsonl"
    running = None
    t0 = time.time()

    with _catch_stop(stop_signals, verbose) as stop, log_path.open("a") as log_file:
        header = {
            "event": "start",
            "n_parameters": n_parameters(model),
            "training_sizes": list(sizes),
            "loss_sizes": [s - 2 * margin for s in sizes],
            "sigma_min": sde.sigma_min,
            "sigma_max": sde.sigma_max,
            "start_step": start_step,
            "resumed_from": str(resume) if resume is not None else None,
            "n_devices": n_devices,
            # Whatever this architecture thinks is worth recording.  Last, so a
            # field it and the common part both name is the model's own.
            **model.log_header(config),
        }
        log_file.write(json.dumps(header) + "\n")
        log_file.flush()

        step = start_step
        for step in range(start_step + 1, cfg.steps + 1):
            # Placed across the devices here rather than left for the constraint
            # inside step_fn to move: this scatters straight from the host
            # instead of landing the whole batch on device 0 and redistributing.
            batch = eqx.filter_shard(jnp.asarray(next(batches)), over_batch)
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

            if (eval_batch is not None and cfg.eval_every
                    and step % cfg.eval_every == 0):
                key, k_eval = jax.random.split(key)
                sigmas = sde.sigma(jnp.linspace(0.0, 1.0, cfg.eval_sigmas))
                per = eval_fn(ema_model, jnp.asarray(eval_batch), k_eval, sigmas)
                n = per.shape[1]
                mean = jnp.mean(per, axis=1)
                # Standard error on that mean, so a bump can be told from the
                # scatter of the patches it was averaged over.
                err = jnp.std(per, axis=1) / jnp.sqrt(n)
                record = {
                    "step": step,
                    "event": "eval",
                    "sigma": [float(s) for s in sigmas],
                    "loss_by_sigma": [float(v) for v in mean],
                    "loss_by_sigma_err": [float(v) for v in err],
                    # What the best Gaussian model of this batch would score.
                    # Above it is definitely underfit; below it means the model
                    # is using structure a covariance cannot express.
                    "gaussian_floor": [
                        float(v) for v in gaussian_loss_floor(
                            jnp.asarray(eval_batch), sigmas)
                    ],
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if on_log is not None:
                    on_log(record)

            if stop:
                # Between steps, where the state is consistent.  `latest` only:
                # this is a pause, so it gets the optimiser state and no
                # numbered checkpoint and no samples -- drawing 64 of those
                # inside a scheduler's grace period is how you lose the chunk.
                save_checkpoint(out / "latest", step, config, model, ema_model,
                                opt_state)
                record = {"step": step, "event": "stopped",
                          "signal": stop.name, "seconds": time.time() - t0}
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if verbose:
                    print(f"  {stop.name} at step {step:,}: saved "
                          f"{out / 'latest'} and stopping. Resume with "
                          f"--resume {out / 'latest'}")
                break

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

    # `final` means finished.  A run stopped by a signal has not finished, and
    # writing it anyway would tell the next chunk -- and scripts/sample.py --
    # that this is the trained model.
    if not stop and step >= cfg.steps:
        save_checkpoint(out / "final", cfg.steps, config, model, ema_model, opt_state)
    return model, ema_model


def _resume(directory, config: Config, model: ScoreModel, optimizer,
            verbose: bool):
    """Restore weights, EMA, optimiser state and step from a checkpoint.

    The optimiser state is the one people forget.  It holds Adam's moments, and
    -- because optax schedules are driven by a count inside it -- the position in
    the warmup and cosine decay.  Starting it fresh re-runs the warmup from zero
    learning rate on a half-trained model, which is not a continuation of
    anything.
    """
    d = Path(directory)
    saved = Config.load(d / "config.json")
    # The *selected* section, not `energy`: with two architectures in one
    # config, comparing the wrong one either misses a real mismatch or refuses a
    # resume over settings the run never used.
    if (saved.architecture != config.architecture
            or saved.model_config != config.model_config):
        raise ValueError(
            f"{d} was trained with a different architecture, so its weights do "
            f"not fit this model:\n  checkpoint: {saved.architecture} "
            f"{saved.model_config}\n  now:        {config.architecture} "
            f"{config.model_config}"
        )
    if saved.train.steps != config.train.steps and verbose:
        print(f"  NOTE: checkpoint had train.steps={saved.train.steps}, now "
              f"{config.train.steps}. The learning-rate schedule decays over "
              f"steps, so changing it mid-run changes the schedule.")
    model, _, step = load_checkpoint(d, which="model")
    ema_model, _, _ = load_checkpoint(d, which="ema")
    opt_state = load_opt_state(
        d, optimizer.init(eqx.filter(model, eqx.is_inexact_array)))
    if verbose:
        print(f"  resumed from {d} at step {step:,} of {config.train.steps:,}")
    return model, ema_model, opt_state, int(step)


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
        from ..diffusion.sampler import sample_scene

        x = sample_scene(
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


def _check_batch(batch: jnp.ndarray, model: ScoreModel) -> None:
    if batch.ndim != 4:
        raise ValueError(f"expected (B, C, H, W) batches, got shape {batch.shape}")
    if batch.shape[1] != model.in_channels:
        raise ValueError(
            f"batch has {batch.shape[1]} channels, model expects "
            f"{model.in_channels}"
        )
    h, w = batch.shape[-2:]
    need = 2 * model.loss_margin + 1
    if min(h, w) < need:
        raise ValueError(
            f"patches are {h}x{w} but the configured loss margin is "
            f"{model.loss_margin} px per side, which leaves nothing to compute "
            f"a loss on. Reduce energy.loss_margin, or feed a larger grid."
        )

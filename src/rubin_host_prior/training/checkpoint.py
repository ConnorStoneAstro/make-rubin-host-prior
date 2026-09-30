"""Checkpointing.

Equinox serialises *leaves* only, so reloading needs a correctly shaped template.
The config is therefore written next to the weights and used to rebuild the
skeleton -- which also means a checkpoint carries everything needed to reproduce
the data transform, without which the prior cannot be used in a forward model.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax

from ..config import Config
from ..nn.score import ScoreModel


def save_checkpoint(
    directory: str | Path,
    step: int,
    config: Config,
    model: ScoreModel,
    ema_model: ScoreModel | None = None,
    opt_state=None,
) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    config.save(d / "config.json")
    (d / "state.json").write_text(json.dumps({"step": int(step)}))
    eqx.tree_serialise_leaves(d / "model.eqx", model)
    if ema_model is not None:
        eqx.tree_serialise_leaves(d / "ema.eqx", ema_model)
    if opt_state is not None:
        eqx.tree_serialise_leaves(d / "opt_state.eqx", opt_state)
    return d


def load_checkpoint(
    directory: str | Path, which: str = "ema"
) -> tuple[ScoreModel, Config, int]:
    """Rebuild a model from a checkpoint directory.

    ``which`` selects ``"ema"`` (default, and what you want for inference) or
    ``"model"`` (the live weights, for resuming training).
    """
    d = Path(directory)
    config = Config.load(d / "config.json")
    step = json.loads((d / "state.json").read_text())["step"]
    # Any key works: every leaf is overwritten by the deserialised values.
    skeleton = config.build_model(jax.random.key(0))
    name = {"ema": "ema.eqx", "model": "model.eqx"}[which]
    path = d / name
    if not path.exists():  # e.g. trained with ema disabled
        path = d / "model.eqx"
    model = eqx.tree_deserialise_leaves(path, skeleton)
    return model, config, step


def load_opt_state(directory: str | Path, template):
    return eqx.tree_deserialise_leaves(Path(directory) / "opt_state.eqx", template)

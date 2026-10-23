"""MLX-only BC and on-policy QAT updates; Qwen features are frozen inputs."""

import json
import math
import os
from pathlib import Path
import shutil
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from .data import file_hash, random_batch, validation_batches, write_json
from .model import load_model, save_model
from .state import bind_identity, resume_exists, resume_directory
from ..score import REWARD_VERSION
from .distribution import SAMPLING_VERSION, gripper_logits


def masked_loss(model, features, state, previous, history, history_valid, target, action_valid):
    prediction = model.actions(features, state, previous, history, history_valid)
    difference = mx.abs(prediction-target)
    huber = mx.where(difference < 1, .5*difference*difference, difference-.5)
    per_action = mx.mean(huber[..., :3], axis=-1) + mx.mean(huber[..., 3:5], axis=-1) + huber[..., 5]
    return mx.sum(per_action*action_valid)/mx.maximum(mx.sum(action_valid), 1)


def apply_gradients(model, optimizer, grads):
    leaves = [value for _, value in tree_flatten(grads)]
    norm = mx.sqrt(sum(mx.sum(value*value) for value in leaves))
    factor = mx.minimum(1., 1./mx.maximum(norm, 1e-9))
    clipped = tree_map(lambda value: value*factor, grads)
    mx.eval(norm)
    if not math.isfinite(float(norm)):
        raise FloatingPointError("Nonfinite MLX gradient")
    optimizer.update(model, clipped)
    return norm


def bc_update(model, optimizer, batch):
    loss, grads = nn.value_and_grad(model, masked_loss)(model, *batch)
    norm = apply_gradients(model, optimizer, grads)
    mx.eval(loss, model.parameters(), optimizer.state)
    if not math.isfinite(float(loss)):
        raise FloatingPointError("Nonfinite MLX BC loss")
    return dict(loss=float(loss), gradient_norm=float(norm))


def validate(model, episodes, batch_size=64):
    snapshot = model.snapshot()
    total, count = 0., 0
    for batch in validation_batches(episodes, model.config, batch_size):
        loss = masked_loss(snapshot, *batch)
        mx.eval(loss)
        n = int(mx.sum(batch[-1]))
        total += float(loss)*n
        count += n
    return total/max(count, 1)


def save_resume(model, optimizer, directory, progress, best_directory=None):
    directory = Path(directory)
    pointer = directory / "current.json"
    previous_name = json.loads(pointer.read_text())["generation"] if pointer.exists() else None
    generations = directory / "generations"
    generation = generations / uuid.uuid4().hex
    generation.mkdir(parents=True)
    try:
        save_model(model, generation / "master", packed=False)
        if best_directory is not None:
            shutil.copytree(best_directory, generation / "best_master")
        state = {name: mx.array(value) for name, value in tree_flatten(optimizer.state)}
        mx.eval(state)
        mx.save_safetensors(str(generation / "optimizer.safetensors"), state)
        write_json(generation / "progress.json", progress)
        manifest = {str(p.relative_to(generation)): file_hash(p) for p in generation.rglob("*") if p.is_file()}
        write_json(generation / "manifest.json", manifest)
        for path in generation.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        write_json(directory / "current.json", dict(generation=generation.name,
                   manifest_sha256=file_hash(generation / "manifest.json")))
        if model.config.backbone_sha256:
            model._backbone_file = generation / "master/qwen_mlp.safetensors"
    except BaseException:
        # Leave even interrupted generations isolated from the committed pointer.
        raise
    # Retain the current and previous generation, bounding checkpoint disk use.
    for old in generations.iterdir():
        if (old.is_dir() and len(old.name) == 32 and all(c in "0123456789abcdef" for c in old.name)
                and old.name not in (generation.name, previous_name)):
            shutil.rmtree(old)


def restore_resume(directory, optimizer):
    directory = resume_directory(directory)
    model, payload = load_model(directory / "master", training=True)
    if payload["packed"]:
        raise ValueError("Resume requires floating master weights")
    optimizer.state = tree_unflatten(list(mx.load(str(directory / "optimizer.safetensors")).items()))
    progress = json.loads((directory / "progress.json").read_text())
    return model, progress


def train_bc(model, train, val, directory, steps=12000, batch_size=32,
             learning_rate=1e-4, eval_every=250, seed=20261008, cache_identity=None, source_identity=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if batch_size % 32 != 0:
        raise ValueError(f"MLX MXFP8 gradient flow requires batch_size % 32 == 0, got {batch_size}")
    optimizer = optim.AdamW(learning_rate=learning_rate)
    rng = np.random.default_rng(seed)
    settings = dict(batch_size=batch_size, learning_rate=learning_rate, eval_every=eval_every, seed=seed)
    bind_identity(directory / "identity.json", dict(cache=cache_identity, source=source_identity, settings=settings, policy=model.config.__dict__),
                  occupied=any((directory / n).exists() for n in ("progress.json", "current.json", "best_master", "result.json")))
    start, best, best_step = 0, math.inf, 0
    original_initial = None
    if resume_exists(directory):
        model, progress = restore_resume(directory, optimizer)
        if progress.get("cache_identity") != cache_identity:
            raise ValueError("BC resume cache changed")
        if progress.get("source_identity") != source_identity:
            raise ValueError("BC resume source checkpoint changed")
        if progress.get("settings") != settings:
            raise ValueError("BC resume hyperparameters changed")
        start, best, best_step = progress["step"], progress["best_val_loss"], progress["best_step"]
        if start > steps:
            raise ValueError("BC step budget is below the saved progress")
        rng.bit_generator.state = progress["rng"]
        original_initial = progress.get("initial_val_loss")
        saved_best, _ = load_model(resume_directory(directory) / "best_master", training=True)
        save_model(saved_best, directory / "best_master", packed=False)
    model.enable_qat()
    model.train()
    measured_initial = validate(model, val)
    initial = measured_initial if original_initial is None else original_initial
    if start == 0:
        best = initial
        save_model(model, directory / "best_master", packed=False)
    for step in range(start, steps):
        batch = random_batch(train, rng, batch_size, model.config)
        metrics = bc_update(model, optimizer, batch)
        model._metadata = {**getattr(model, "_metadata", {}), "fp8_training_completed": True}
        if (step+1) % eval_every == 0 or step+1 == steps:
            value = validate(model, val)
            if value < best:
                best, best_step = value, step+1
                save_model(model, directory / "best_master", packed=False)
            record = dict(stage="mlx_bc", step=step+1, val_loss=value, best_val_loss=best, **metrics)
            with (directory / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            save_resume(model, optimizer, directory, dict(step=step+1, best_val_loss=best,
                        best_step=best_step, rng=rng.bit_generator.state, cache_identity=cache_identity,
                        source_identity=source_identity, settings=settings, initial_val_loss=initial),
                        best_directory=directory / "best_master")
    best_model, _ = load_model(directory / "best_master", training=True)
    save_model(best_model, directory / "model", packed=True,
               metadata=dict(bc_steps=steps, selected_step=best_step, qat_during_bc=True))
    result = dict(complete=True, stage="mlx_bc", steps=steps, best_step=best_step,
                  initial_val_loss=initial, best_val_loss=best,
                  train_episodes=len(train), val_episodes=len(val),
                  qat_layers=list(best_model.linear_names), qat_mode="mxfp8", base_quantized=False,
                  task_success_verified=False)
    write_json(directory / "result.json", result)
    return best_model, result


def action_log_probability(actions, latent, closed, noise_scale=.5):
    xyz = mx.arctanh(mx.clip(actions[..., :3], -.999, .999))
    yaw = mx.arctan2(actions[..., 3:4], actions[..., 4:5])
    means = mx.concatenate((xyz, yaw), axis=-1)
    scales = mx.array([.10, .10, .06, .08])*noise_scale
    normal = -.5*((latent-means)/scales)**2-mx.log(scales)-.5*math.log(2*math.pi)
    logits = gripper_logits(actions[..., 5], noise_scale, mx)
    bernoulli = closed*logits-mx.logaddexp(mx.zeros_like(logits), logits)
    return mx.sum(normal, axis=-1)+bernoulli


def reinforce_update(model, optimizer, experiences, noise_scale=.5):
    """Replay exact packed-policy observations, history and executed prefixes."""
    if any(e["result"].get("reward_version") != REWARD_VERSION or
           e["result"].get("sampling_version") != SAMPLING_VERSION for e in experiences):
        raise ValueError({"experience_contract": (REWARD_VERSION, SAMPLING_VERSION)})
    rewards = np.asarray([episode["result"]["training_reward"] for episode in experiences], np.float32)
    if not np.isfinite(rewards).all():
        raise ValueError({"nonfinite_training_reward": True})
    if len(rewards) < 2:
        raise ValueError("REINFORCE requires at least two episodes")
    if rewards.std() < 1e-6:
        return dict(update_applied=False, reason="identical_rewards", mean_reward=float(rewards.mean()))
    advantages = (rewards-rewards.mean())/max(rewards.std(), 1e-4)
    replay_error = 0.
    for episode in experiences:
        for decision in episode["decisions"]:
            inputs = [mx.array(decision[name]) for name in ("features", "state", "previous", "history", "valid")]
            prediction = model.actions(*inputs)
            error = mx.max(mx.abs(prediction-mx.array(decision["reference"])))
            mx.eval(error)
            replay_error = max(replay_error, float(error))
    if replay_error > 1e-3:
        raise RuntimeError(f"FP8 on-policy replay mismatch: {replay_error}")

    def loss_fn(policy):
        losses = []
        for advantage, episode in zip(advantages, experiences, strict=True):
            terms = []
            for decision in episode["decisions"]:
                n = decision["executed_steps"]
                if n == 0:
                    continue
                inputs = [mx.array(decision[name]) for name in ("features", "state", "previous", "history", "valid")]
                output = policy.actions(*inputs)
                terms.append(mx.sum(action_log_probability(output[:, :n],
                    mx.array(decision["latent"])[:, :n], mx.array(decision["closed"])[:, :n], noise_scale)))
            losses.append(-float(advantage)*sum(terms))
        return sum(losses)/len(losses)

    mx.eval(model.pose_projection.core.weight)
    before = np.asarray(model.pose_projection.core.weight).copy()
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    norm = apply_gradients(model, optimizer, grads)
    mx.eval(loss, model.parameters(), optimizer.state)
    change = float(np.max(np.abs(np.asarray(model.pose_projection.core.weight)-before)))
    if not math.isfinite(float(loss)) or change == 0:
        raise RuntimeError("RL update must be finite and change policy weights")
    return dict(update_applied=True, method="on_policy_REINFORCE", loss=float(loss),
                gradient_norm=float(norm), weight_change=change, replay_error=replay_error,
                episodes=len(rewards), mean_reward=float(rewards.mean()),
                successes=sum(bool(e["result"]["success"]) for e in experiences))

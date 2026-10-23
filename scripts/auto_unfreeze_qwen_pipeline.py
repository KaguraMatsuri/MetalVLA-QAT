"""Fully automated Qwen unfreezing pipeline: train → evaluate → record results.

This script handles the complete workflow:
1. Unfreeze last N Qwen decoder MLP layers
2. Train with BC for specified steps
3. Automatically evaluate on 10 visible MuJoCo episodes
4. Save results and update PROJECT_HISTORY
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

# The official launcher sets MJPYTHON_BIN and initializes macOS's UI thread.
# Switch before loading models or creating any training output.
if __name__ == "__main__" and sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        mjpython = Path(sys.base_prefix) / "bin/mjpython"
    os.execv(sys.executable, [sys.executable, str(mjpython), *sys.orig_argv[1:]])

if __name__ == "__main__":
    import importlib.util
    if importlib.util.find_spec("mlx") is None:
        raise ModuleNotFoundError("mlx")

import numpy as np
import torch
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ur_pick_place_autoloop_v1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from metalvla_qat.qwen_pose_vla import QwenPoseVLA, pose_loss
from metalvla_qat.qwen_vla import BASE_MODEL, BASE_REVISION
from metalvla_qat.rgb import preprocess_rgb
from ur_pick_place import URTask, PERIOD, WORKSPACE_LOW, WORKSPACE_HIGH, INSTRUCTION, SCHEMA
import ur_pose_worker as worker


def vocabulary_for(text):
    words = list(dict.fromkeys(text.split()))
    return {word: index + 1 for index, word in enumerate(words)}


def unfrozen_backbone_prefixes(model, unfreeze_layers):
    total_layers = len(model.encoder.backbone.language_model.layers)
    return tuple(
        f"encoder.backbone.language_model.layers.{index}.mlp."
        for index in range(total_layers - unfreeze_layers, total_layers)
    )


def checkpoint_adapter(model, unfreeze_layers):
    """Persist task weights and every selected decoder MLP on the CPU."""
    prefixes = unfrozen_backbone_prefixes(model, unfreeze_layers)
    return {
        name: value.detach().cpu() for name, value in model.state_dict().items()
        if not name.startswith("encoder.backbone.") or name.startswith(prefixes)
    }


def load_unfrozen_model(checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = QwenPoseVLA(**saved["model_config"])
    prefixes = unfrozen_backbone_prefixes(model, saved["unfrozen_layers"])
    # Only unchanged backbone weights come from the pinned pretrained model.
    # Strict loading requires all task weights and selected MLP weights on disk.
    state = {
        name: value for name, value in model.state_dict().items()
        if name.startswith("encoder.backbone.") and not name.startswith(prefixes)
    }
    state.update({name: value for name, value in saved["adapter"].items()
                  if not name.startswith(("encoder.feature_norm.", "encoder.state_encoder.", "encoder.action_head."))})
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), saved


def load_visible_policy(checkpoint_path, model_directory, device="mps"):
    from metalvla_qat.mac.model import import_torch_checkpoint
    from metalvla_qat.mac.runtime import VisiblePolicy
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_directory = Path(model_directory)
    if not (model_directory / "config.json").exists():
        import_torch_checkpoint(checkpoint_path, model_directory)
    else:
        payload = json.loads((model_directory / "config.json").read_text())
        if payload["metadata"]["source_sha256"] != sha256_file(Path(checkpoint_path)):
            raise ValueError({"source_checkpoint": str(checkpoint_path)})
    return VisiblePolicy(model_directory, device=device), saved


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(path, split):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("split") == split]
    if not selected or any(row.get("schema") != SCHEMA or not row.get("success") for row in selected):
        raise ValueError(f"Invalid {split} split in {path}")
    return selected


def load_batch(rows, device, model, vocabulary):
    """Load a batch of episodes for training."""
    raw_images, states, previous, targets = [], [], [], []
    horizon = model.config.horizon
    for row in rows:
        path = DATA / row["file"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Episode mismatch: {path}")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata.get("schema") != SCHEMA or metadata.get("success") is not True:
                raise ValueError(f"Invalid episode: {path}")
            desired = saved["desired"]
            action = np.empty((horizon, 5), dtype=np.float32)
            count = min(horizon, len(desired))
            action[:count] = desired[:count]
            action[count:] = desired[count - 1]
            raw_images.append(saved["images"][0].copy())
            states.append(saved["state"][0])
            previous.append(saved["previous"][0])
            targets.append(action)
    images = preprocess_rgb(torch.from_numpy(np.stack(raw_images)), model.config.image_size).to(device)
    state = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)
    prev = torch.from_numpy(np.asarray(previous, dtype=np.float32)).to(device)
    desired = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    target = model.targets(desired)
    tokens = torch.arange(1, len(vocabulary) + 1, device=device)[None].expand(len(rows), -1)
    return images, tokens, state, prev, target


def train_unfrozen_qwen(output, steps, batch_size, lr, unfreeze_layers, device, seed,
                        base_model=BASE_MODEL, base_revision=BASE_REVISION,
                        backbone_dtype="float32", eval_batch_size=32, val_samples=32):
    """Train Qwen with unfrozen decoder layers."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: TRAINING UNFROZEN QWEN")
    print(f"{'='*70}")

    manifest = DATA / "raw/manifest.jsonl"
    train_rows = manifest_rows(manifest, "train")
    val_rows = manifest_rows(manifest, "val")
    if (len(train_rows), len(val_rows)) != (2592, 288):
        raise ValueError(f"Unexpected split: train={len(train_rows)}, val={len(val_rows)}")

    vocabulary = vocabulary_for(INSTRUCTION)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Create model
    model = QwenPoseVLA(vocabulary, workspace_low=WORKSPACE_LOW,
                        workspace_high=WORKSPACE_HIGH, base_model=base_model,
                        base_revision=base_revision, backbone_dtype=backbone_dtype).to(device)

    # Unfreeze strategy
    total_layers = len(model.encoder.backbone.language_model.layers)
    model.encoder.backbone.requires_grad_(False)

    if unfreeze_layers > 0:
        for i in range(total_layers - unfreeze_layers, total_layers):
            layer = model.encoder.backbone.language_model.layers[i]
            for param in layer.mlp.parameters():
                param.requires_grad = True
            print(f"✓ Unfrozen: backbone.language_model.layers.{i}.mlp")

    # Always train task head
    model.feature_norm.requires_grad_(True)
    model.state_encoder.requires_grad_(True)
    model.action_head.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: trainable={trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"Unfrozen layers: {unfreeze_layers} | Steps: {steps} | Batch: {batch_size} | LR: {lr}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )

    model.train()
    if unfreeze_layers > 0:
        model.encoder.backbone.eval()  # Keep BatchNorm in eval mode

    step = 0
    best_val_loss = float("inf")
    training_log = []
    start_time = time.perf_counter()

    print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    while step < steps:
        batch_idx = rng.choice(len(train_rows), batch_size, replace=False)
        batch_rows = [train_rows[i] for i in batch_idx]

        images, tokens, state, previous, target = load_batch(batch_rows, device, model, vocabulary)

        pred = model.actions_from_features(
            model.features(images, tokens), state, previous
        )
        loss = pose_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        step += 1

        if step % 100 == 0 or step == steps:
            elapsed = time.perf_counter() - start_time
            steps_per_sec = step / elapsed
            eta_sec = (steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"[{step:4d}/{steps}] loss={loss.item():.6f} | grad={grad_norm.item():.4f} | "
                  f"{elapsed:.1f}s | ETA: {eta_sec/60:.1f}min")

        # Validation
        if step % 200 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                eval_idx = rng.choice(len(val_rows), min(val_samples, len(val_rows)), replace=False)
                eval_rows = [val_rows[i] for i in eval_idx]
                val_total = 0.
                for start in range(0, len(eval_rows), eval_batch_size):
                    rows = eval_rows[start:start + eval_batch_size]
                    images, tokens, state, previous, target = load_batch(rows, device, model, vocabulary)
                    pred = model.actions_from_features(model.features(images, tokens), state, previous)
                    val_total += float(pose_loss(pred, target).item()) * len(rows)
                val_loss = val_total / len(eval_rows)

            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "grad_norm": float(grad_norm.item()),
            }
            training_log.append(record)
            print(f"          val_loss={val_loss:.6f}", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model_config": model.model_config,
                    "adapter": checkpoint_adapter(model, unfreeze_layers),
                    "policy_kind": "qwen_ur_pose_v1_unfrozen",
                    "unfrozen_layers": unfreeze_layers,
                    "training_steps": step,
                    "best_val_loss": best_val_loss,
                    "trainable_params": trainable,
                    "conversion_gate_passed": False,
                }
                torch.save(checkpoint, output / "model_best.pt")
                print(" ← BEST", end="")
            print()

            model.train()
            if unfreeze_layers > 0:
                model.encoder.backbone.eval()

    # Save final
    final_checkpoint = {
        "model_config": model.model_config,
        "adapter": checkpoint_adapter(model, unfreeze_layers),
        "policy_kind": "qwen_ur_pose_v1_unfrozen",
        "unfrozen_layers": unfreeze_layers,
        "training_steps": step,
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "trainable_params": trainable,
        "conversion_gate_passed": False,
    }
    torch.save(final_checkpoint, output / "model_final.pt")

    with open(output / "training.json", "w") as f:
        json.dump({"config": {
            "steps": steps, "batch_size": batch_size, "lr": lr,
            "unfreeze_layers": unfreeze_layers, "device": device, "seed": seed,
            "base_model": base_model, "base_revision": base_revision,
            "backbone_dtype": backbone_dtype, "eval_batch_size": eval_batch_size,
            "val_samples": val_samples,
        }, "training_log": training_log}, f, indent=2)

    total_time = time.perf_counter() - start_time
    print(f"\n✓ Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Final val_loss: {val_loss:.6f}")

    return output / "model_best.pt", best_val_loss


def evaluate_mujoco(checkpoint_path, output, device, seed):
    """Evaluate on 10 visible MuJoCo episodes."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: MUJOCO EVALUATION (10 episodes)")
    print(f"{'='*70}\n")

    model, saved = load_visible_policy(checkpoint_path, output / "mlx_model", device=device)

    task = URTask(render=True)
    from types import SimpleNamespace
    rollout_args = SimpleNamespace(
        run=output, min_free_gb=5., max_decisions=3, grasp_retries=2,
        precision_reward=True, rl_noise_scale=1., round=0, seed=seed,
    )

    results = []
    eval_dir = output / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    try:
        with mujoco.viewer.launch_passive(task.model, task.data) as viewer:
            for index in range(10):
                region = index % 9
                episode_seed = seed + index
                original_execute = task.execute

                def paced_execute(xyz, yaw, closed):
                    original_execute(xyz, yaw, closed)
                    viewer.sync()
                    time.sleep(PERIOD)

                task.execute = paced_execute
                try:
                    start = time.perf_counter()
                    result, _ = worker.rollout(
                        rollout_args, task, model, episode_seed, region, index,
                    )
                    elapsed = time.perf_counter() - start
                finally:
                    task.execute = original_execute

                results.append(result)
                worker.save_json(eval_dir / f"episode_{index+1:03d}.json",
                                dict(result=result, trace=task.trace))

                status = "✓" if result["success"] else "✗"
                print(f"[{index+1:2d}/10] {status} region={region+1} seed={episode_seed} | "
                      f"score={result['score']:.3f} | {result['failure_type']:12s} | "
                      f"retries={result.get('grasp_retries',0)} | {elapsed:.1f}s")

            successes = sum(r["success"] for r in results)
            worker.save_json(eval_dir / "result.json", dict(
                episodes=10,
                successes=successes,
                success_rate=f"{successes}/10 ({100*successes/10:.0f}%)",
                results=results,
                checkpoint=str(checkpoint_path),
                unfrozen_layers=saved.get("unfrozen_layers"),
                trainable_params=saved.get("trainable_params"),
                device=device,
            ))

            print(f"\n{'='*70}")
            print(f"RESULT: {successes}/10 successful ({100*successes/10:.0f}%)")
            print(f"{'='*70}\n")

            print("MuJoCo viewer will remain open for 10 seconds...")
            for i in range(10):
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(1)

            return successes, results
    finally:
        task.close()
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--unfreeze-layers", type=int, choices=(2,), default=2,
                        help="Number of last Qwen decoder layers to unfreeze")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--eval-seed", type=int, default=21270001,
                        help="Seed for evaluation episodes")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--backbone-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=32)
    args = parser.parse_args()

    if not 0 <= args.unfreeze_layers <= 4:
        parser.error("unfreeze-layers must be 0-4")
    if min(args.steps, args.batch_size, args.eval_batch_size, args.val_samples) < 1:
        raise ValueError(vars(args))

    output = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device

    pipeline_start = time.perf_counter()

    print(f"\n{'#'*70}")
    print(f"# AUTOMATED QWEN UNFREEZING PIPELINE")
    print(f"#")
    print(f"# Output: {output}")
    print(f"# Unfreeze layers: {args.unfreeze_layers}")
    print(f"# Training: {args.steps} steps, batch={args.batch_size}, lr={args.lr}")
    print(f"# Device: {device}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # Phase 1: Training
    checkpoint_path, best_val_loss = train_unfrozen_qwen(
        output, args.steps, args.batch_size, args.lr,
        args.unfreeze_layers, device, args.seed,
        base_model=args.base_model, base_revision=args.base_revision,
        backbone_dtype=args.backbone_dtype, eval_batch_size=args.eval_batch_size,
        val_samples=args.val_samples,
    )

    # Phase 2: Evaluation
    successes, results = evaluate_mujoco(checkpoint_path, output, device, args.eval_seed)

    # Save summary
    pipeline_time = time.perf_counter() - pipeline_start
    summary = {
        "pipeline": "automated_qwen_unfreezing",
        "output_dir": str(output),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "unfreeze_layers": args.unfreeze_layers,
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": device,
            "train_seed": args.seed,
            "eval_seed": args.eval_seed,
        },
        "results": {
            "best_val_loss": best_val_loss,
            "mujoco_successes": successes,
            "mujoco_episodes": 10,
            "success_rate": f"{successes}/10 ({100*successes/10:.0f}%)",
        },
        "pipeline_time_seconds": pipeline_time,
    }

    with open(output / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# PIPELINE COMPLETE")
    print(f"#")
    print(f"# Total time: {pipeline_time:.1f}s ({pipeline_time/60:.1f} min)")
    print(f"# Training: best_val_loss={best_val_loss:.6f}")
    print(f"# Evaluation: {successes}/10 successful ({100*successes/10:.0f}%)")
    print(f"#")
    print(f"# Results saved to: {output}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    from metalvla_qat.mac.state import GPU_LOCK, exclusive_lock
    with exclusive_lock(GPU_LOCK):
        main()

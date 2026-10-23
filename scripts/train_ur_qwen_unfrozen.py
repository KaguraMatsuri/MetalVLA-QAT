"""Unfreeze last N Qwen decoder MLP layers and fine-tune on UR pick-place task.

This script starts from existing frozen-Qwen data and selectively unfreezes
the last decoder MLP layers to let visual features adapt to spatial reasoning.
Attention weights stay frozen to preserve language knowledge.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ur_pick_place_autoloop_v1"
sys.path.insert(0, str(ROOT / "src"))

from metalvla_qat.qwen_pose_vla import QwenPoseVLA, pose_loss
from metalvla_qat.qwen_vla import BASE_MODEL, BASE_REVISION
from metalvla_qat.rgb import preprocess_rgb
from auto_unfreeze_qwen_pipeline import checkpoint_adapter
from ur_pick_place import SCHEMA, INSTRUCTION, WORKSPACE_LOW, WORKSPACE_HIGH


def vocabulary_for(text):
    words = list(dict.fromkeys(text.split()))
    return {word: index + 1 for index, word in enumerate(words)}


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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000,
                        help="Training steps (lower than frozen BC)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Lower learning rate for unfreezing")
    parser.add_argument("--unfreeze-layers", type=int, choices=(2,), default=2,
                        help="Number of last decoder layers to unfreeze")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--backbone-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=32)
    args = parser.parse_args()

    if not 0 <= args.unfreeze_layers <= 4:
        parser.error("unfreeze-layers must be 0-4")
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batch_size, args.val_samples) < 1:
        parser.error("steps and batch-size must be positive")

    output = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = DATA / "raw/manifest.jsonl"
    train_rows = manifest_rows(manifest, "train")
    val_rows = manifest_rows(manifest, "val")
    if (len(train_rows), len(val_rows)) != (2592, 288):
        raise ValueError(f"Unexpected split: train={len(train_rows)}, val={len(val_rows)}")

    vocabulary = vocabulary_for(INSTRUCTION)
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Create model
    model = QwenPoseVLA(vocabulary, workspace_low=WORKSPACE_LOW,
                        workspace_high=WORKSPACE_HIGH, base_model=args.base_model,
                        base_revision=args.base_revision, backbone_dtype=args.backbone_dtype).to(device)

    # Unfreeze strategy
    total_layers = len(model.encoder.backbone.language_model.layers)
    model.encoder.backbone.requires_grad_(False)  # Freeze all first

    if args.unfreeze_layers > 0:
        for i in range(total_layers - args.unfreeze_layers, total_layers):
            layer = model.encoder.backbone.language_model.layers[i]
            # Unfreeze MLP only, keep attention frozen
            for param in layer.mlp.parameters():
                param.requires_grad = True
            print(f"Unfrozen: backbone.language_model.layers.{i}.mlp")

    # Always train task head
    model.feature_norm.requires_grad_(True)
    model.state_encoder.requires_grad_(True)
    model.action_head.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: trainable={trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"Unfrozen Qwen decoder layers: {args.unfreeze_layers}")
    print(f"Training: {args.steps} steps, batch_size={args.batch_size}, lr={args.lr}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )

    model.train()
    # Keep Qwen eval mode for BatchNorm, but allow gradients
    if args.unfreeze_layers > 0:
        model.encoder.backbone.eval()

    step = 0
    best_val_loss = float("inf")
    training_log = []
    start_time = time.perf_counter()

    while step < args.steps:
        # Sample random batch
        batch_idx = rng.choice(len(train_rows), args.batch_size, replace=False)
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

        if step % 50 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start_time
            print(f"Step {step}/{args.steps} | train_loss={loss.item():.6f} | "
                  f"grad_norm={grad_norm.item():.4f} | {elapsed:.1f}s")

        # Validation
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                # Eval on subset to save time
                eval_idx = rng.choice(len(val_rows), min(args.val_samples, len(val_rows)), replace=False)
                eval_rows = [val_rows[i] for i in eval_idx]
                for start in range(0, len(eval_rows), args.eval_batch_size):
                    rows = eval_rows[start:start + args.eval_batch_size]
                    images, tokens, state, previous, target = load_batch(rows, device, model, vocabulary)
                    pred = model.actions_from_features(model.features(images, tokens), state, previous)
                    val_losses.append(float(pose_loss(pred, target).item()) * len(rows))
                val_loss = sum(val_losses) / len(eval_rows)

            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "grad_norm": float(grad_norm.item()),
            }
            training_log.append(record)
            print(f"  Validation: val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model_config": model.model_config,
                    "adapter": checkpoint_adapter(model, args.unfreeze_layers),
                    "policy_kind": "qwen_ur_pose_v1_unfrozen",
                    "unfrozen_layers": args.unfreeze_layers,
                    "training_steps": step,
                    "best_val_loss": best_val_loss,
                    "trainable_params": trainable,
                    "conversion_gate_passed": False,
                }
                torch.save(checkpoint, output / "model_best.pt")
                print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.6f})")

            model.train()
            if args.unfreeze_layers > 0:
                model.encoder.backbone.eval()

    # Save final
    final_checkpoint = {
        "model_config": model.model_config,
        "adapter": checkpoint_adapter(model, args.unfreeze_layers),
        "policy_kind": "qwen_ur_pose_v1_unfrozen",
        "unfrozen_layers": args.unfreeze_layers,
        "training_steps": step,
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "trainable_params": trainable,
        "conversion_gate_passed": False,
    }
    torch.save(final_checkpoint, output / "model_final.pt")

    with open(output / "training.json", "w") as f:
        config = {**vars(args), "output_dir": str(args.output_dir)}
        json.dump({"config": config, "training_log": training_log}, f, indent=2)

    total_time = time.perf_counter() - start_time
    print(f"\n✓ Training complete in {total_time:.1f}s")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {output}")


if __name__ == "__main__":
    from metalvla_qat.mac.state import GPU_LOCK, exclusive_lock
    with exclusive_lock(GPU_LOCK):
        main()

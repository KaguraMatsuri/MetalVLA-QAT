"""Quick 10-episode visual evaluation for unfrozen Qwen checkpoint."""

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ur_pose_worker as worker
from auto_unfreeze_qwen_pipeline import load_visible_policy
from ur_pick_place import URTask, PERIOD
from metalvla_qat.mac.runtime import VisiblePolicy
from metalvla_qat.mac.data import file_hash, write_json
from metalvla_qat.mac.state import GPU_LOCK, bind_identity, exclusive_lock, model_identity
from metalvla_qat.score import REWARD_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21270001)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--close-after", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with exclusive_lock(GPU_LOCK):
        return evaluate(args)


def evaluate(args):
    selected = args.model if args.model is not None else args.checkpoint
    checkpoint_path = selected if selected.is_absolute() else ROOT / selected
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=args.resume)
    identity = bind_identity(output / "identity.json",
        dict(schema=2, reward_version=REWARD_VERSION, source=model_identity(checkpoint_path) if args.model is not None else file_hash(checkpoint_path),
             seed=args.seed, episodes=10, device=args.device, region_modulo=9, max_decisions=3, retries=2),
        occupied=any(output.glob("episode_*.json")) or (output / "result.json").exists())
    progress_path = output / "result.json"
    prior = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    results = prior.get("results", [])
    if prior and prior.get("identity") != identity:
        raise ValueError({"evaluation_identity_changed": str(progress_path)})
    if len(results) > 10 or any(r.get("seed") != args.seed+i or r.get("region") != i % 9 or r.get("ordinal") != i for i, r in enumerate(results)):
        raise ValueError({"evaluation_sequence_changed": str(progress_path)})
    if len(results) == 10:
        return prior

    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device

    print(f"Loading unfrozen Qwen checkpoint from {checkpoint_path}")
    if args.model is not None:
        model = VisiblePolicy(checkpoint_path, device=device)
        saved = {"unfrozen_layers": 2}
    else:
        model, saved = load_visible_policy(checkpoint_path, output / "mlx_model", device=device)

    unfrozen_layers = saved.get("unfrozen_layers", "unknown")
    trainable_params = saved.get("trainable_params", "unknown")
    print(f"Unfrozen Qwen layers: {unfrozen_layers}")
    print(f"Trainable parameters: {trainable_params:,}" if isinstance(trainable_params, int) else trainable_params)
    print(f"Device: {device}")

    task = URTask(render=True)
    from types import SimpleNamespace
    rollout_args = SimpleNamespace(
        run=output, min_free_gb=5., max_decisions=3, grasp_retries=2,
        precision_reward=True, rl_noise_scale=1., round=0, seed=args.seed,
    )

    try:
        with mujoco.viewer.launch_passive(task.model, task.data) as viewer:
            print(f"\nRunning 10 episodes with visible MuJoCo viewer (realtime-factor=1)")
            print(f"Unfrozen Qwen + H=130 one-shot prediction + up to 2 retries\n")

            for index in range(len(results), 10):
                region = index % 9
                seed = args.seed + index
                original_execute = task.execute

                def paced_execute(xyz, yaw, closed):
                    original_execute(xyz, yaw, closed)
                    viewer.sync()
                    time.sleep(PERIOD)  # realtime

                task.execute = paced_execute
                try:
                    start = time.perf_counter()
                    result, _ = worker.rollout(
                        rollout_args, task, model, seed, region, index,
                    )
                    elapsed = time.perf_counter() - start
                finally:
                    task.execute = original_execute

                result.update(seed=seed, region=region, ordinal=index)
                results.append(result)
                worker.save_json(output / f"episode_{index+1:03d}.json",
                                dict(result=result, trace=task.trace))

                write_json(progress_path, dict(complete=len(results) == 10, identity=identity,
                    episodes=len(results), successes=sum(r["success"] for r in results), results=results,
                    checkpoint=str(checkpoint_path), unfrozen_layers=unfrozen_layers, device=device))

                status = "✓ SUCCESS" if result["success"] else f"✗ {result['failure_type']}"
                print(f"[{index+1}/10] region={region+1} seed={seed} | {status} | "
                      f"score={result['score']:.3f} | retries={result.get('grasp_retries',0)} | "
                      f"{elapsed:.1f}s")

            successes = sum(r["success"] for r in results)
            worker.save_json(output / "result.json", dict(
                complete=True, identity=identity, episodes=10, successes=successes,
                success_rate=f"{successes}/10 ({100*successes/10:.0f}%)",
                results=results,
                checkpoint=str(checkpoint_path),
                unfrozen_layers=unfrozen_layers,
                device=device,
            ))

            print(f"\n{'='*60}")
            print(f"FINAL RESULT: {successes}/10 successful ({100*successes/10:.0f}%)")
            print(f"{'='*60}")
            if args.close_after is None:
                print("\nClose the MuJoCo window to exit.")
            deadline = None if args.close_after is None else time.monotonic() + args.close_after

            while viewer.is_running() and (deadline is None or time.monotonic() < deadline):
                viewer.sync()
                time.sleep(.1)
    finally:
        task.close()
        model.close()


if __name__ == "__main__":
    main()

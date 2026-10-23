"""Same-observation official-Qwen runtime profiling; no benchmark while training."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .data import load_episode, read_manifest, write_json
from .encoder import EncoderConfig, OfficialQwenEncoder, messages_from_pixels


def old_jobs_active():
    result = subprocess.run(["ps", "-axo", "pid=,comm=,args="], capture_output=True, text=True, check=True)
    active = []
    names = ("scripts/train_ur_qwen_unfrozen.py", "scripts/auto_unfreeze_qwen_pipeline.py",
             "scripts/mac_vla.py bc", "scripts/mac_vla.py rl",
             "scripts/mac_vla.py cache", "scripts/mac_vla.py evaluate")
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        executable = Path(fields[1]).name
        if (executable.startswith("python") or executable == "mjpython") and any(name in fields[2] for name in names):
            active.append(line.strip())
    return active


class RoundTripEncoder(OfficialQwenEncoder):
    """Reproduce the previous MPS -> CPU image bounce as the performance baseline."""
    def prepare(self, raw_rgb, instructions):
        import torch
        from ..rgb import preprocess_rgb
        started = time.perf_counter()
        images = preprocess_rgb(torch.from_numpy(raw_rgb.copy()), self.config.image_size).to(self.device)
        pixels = images.cpu().permute(0, 1, 3, 4, 2).mul(255).round().to(torch.uint8).numpy()
        if isinstance(instructions, str):
            instructions = [instructions]*len(pixels)
        messages = messages_from_pixels(pixels, instructions)
        tensors = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
        return tensors, time.perf_counter()-started


def output_gate(policy, reference, candidate, state, previous):
    import mlx.core as mx
    history, valid = policy.empty_history(len(reference))
    a = policy.actions(mx.array(reference), mx.array(state), mx.array(previous), history, valid)
    b = policy.actions(mx.array(candidate), mx.array(state), mx.array(previous), history, valid)
    mx.eval(a, b)
    xyz_a, yaw_a, grip_a = policy.decode(a)
    xyz_b, yaw_b, grip_b = policy.decode(b)
    distance = np.linalg.norm(xyz_a-xyz_b, axis=-1)
    yaw_delta = np.abs(np.arctan2(np.sin(yaw_a-yaw_b), np.cos(yaw_a-yaw_b)))
    far = np.abs(np.asarray(a)[..., 5]) >= .05
    stable_flips = int(np.sum((grip_a != grip_b)&far))
    relative_feature_mae = float(np.abs(candidate-reference).mean()/max(np.abs(reference).mean(), 1e-6))
    result = dict(position_max_m=float(distance.max()), yaw_max_deg=float(np.rad2deg(yaw_delta.max())),
                  stable_gripper_flips=stable_flips, all_gripper_flips=int(np.sum(grip_a != grip_b)),
                  relative_feature_mae=relative_feature_mae)
    result["passed"] = (result["position_max_m"] <= .003 and result["yaw_max_deg"] <= 1.
                        and stable_flips == 0 and relative_feature_mae <= .02)
    return result


def benchmark(policy, dataset_root, output, repeats=3):
    import mlx.core as mx
    if old_jobs_active():
        raise RuntimeError("Existing GPU job active; performance benchmark must run after it finishes")
    rows, _ = read_manifest(dataset_root)
    selected = {}
    for row in rows:
        if row["split"] == "val":
            selected.setdefault(row["region"], row)
    episodes = [load_episode(dataset_root, row, policy.config) for row in selected.values()]
    raw = np.stack([e["raw"][0] for e in episodes])
    state = np.stack([e["state"][0] for e in episodes])
    previous = np.stack([e["previous"][0] for e in episodes])
    instruction = "pick blue cube and place on red target"
    config = policy.config
    profiles = [(dtype, attention) for dtype in ("float32", "float16", "bfloat16") for attention in ("eager", "sdpa")]
    results, reference = [], None
    for label, dtype, attention in [("legacy_roundtrip", "float32", "eager")]+[
            (f"{dtype}_{attention}", dtype, attention) for dtype, attention in profiles]:
        encoder = None
        try:
            ec = EncoderConfig(config.base_model, config.base_revision, dtype, attention, config.image_size,
                               str(policy._backbone_file), config.backbone_sha256)
            cls = RoundTripEncoder if label == "legacy_roundtrip" else OfficialQwenEncoder
            encoder = cls(ec)
            encoder.encode(raw[:1], instruction)
            outputs, latencies = [], []
            for _ in range(repeats):
                collected = []
                for index in range(len(raw)):
                    started = time.perf_counter()
                    feature = encoder.encode(raw[index:index+1], instruction)
                    h, v = policy.empty_history(1)
                    action = policy.actions(mx.array(feature), mx.array(state[index:index+1]),
                                            mx.array(previous[index:index+1]), h, v)
                    mx.eval(action)
                    latencies.append(time.perf_counter()-started)
                    collected.append(feature)
                outputs.append(np.concatenate(collected))
            actual = outputs[-1]
            if reference is None:
                reference = actual
            gate = output_gate(policy, reference, actual, state, previous)
            throughput = {}
            for batch_size in (1, 4, 8, 16):
                # Nine region probes are insufficient for a real batch of 16.
                # Repeat probes to whole batches so labels match measured work.
                count = math.ceil(len(raw)/batch_size)*batch_size
                probes = raw[np.arange(count) % len(raw)]
                started = time.perf_counter()
                for _ in range(repeats):
                    for index in range(0, count, batch_size):
                        encoded = encoder.encode(probes[index:index+batch_size], instruction)
                        if encoded.shape[1] != config.feature_dim:
                            raise ValueError("Changed feature shape")
                elapsed = time.perf_counter()-started
                throughput[str(batch_size)] = count*repeats/elapsed
            import torch
            result = dict(label=label, profile=asdict(ec), numerical_gate=gate,
                          image_to_action_median_s=float(np.median(latencies)),
                          image_to_action_p95_s=float(np.quantile(latencies, .95)),
                          features_per_second=throughput,
                          mps_driver_bytes=torch.mps.driver_allocated_memory(),
                          mlx_peak_bytes=mx.get_peak_memory())
            results.append(result)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            results.append(dict(label=label, error=f"{type(error).__name__}: {error}"))
            if label == "legacy_roundtrip":
                raise
        finally:
            if encoder is not None:
                encoder.close()
        write_json(Path(output) / "progress.json", dict(profiles=results))
    candidates = [r for r in results[1:] if r.get("numerical_gate", {}).get("passed")]
    if not candidates:
        raise RuntimeError("No optimized official Qwen profile passed numerical gates")
    selected = min(candidates, key=lambda r: r["image_to_action_p95_s"])
    batch = max(selected["features_per_second"], key=selected["features_per_second"].get)
    baseline = results[0]
    speedup = selected["features_per_second"][batch]/max(baseline["features_per_second"].values())
    report = dict(profiles=results, selected=selected["profile"], feature_batch_size=int(batch),
                  feature_throughput_speedup=speedup, latency_target_passed=selected["image_to_action_p95_s"]<=1.,
                  throughput_target_passed=speedup>=2., closed_loop_gate_passed=False,
                  default_switch_allowed=False, hardware="Apple Silicon", qat_mode="mxfp8")
    write_json(Path(output) / "result.json", report)
    return report

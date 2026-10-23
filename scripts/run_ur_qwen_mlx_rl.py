"""Run the existing MLX RL command, then the existing visible evaluator."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metalvla_qat.mac.data import file_hash, write_json
from metalvla_qat.mac.state import bind_identity, exclusive_lock, model_identity
from metalvla_qat.score import REWARD_VERSION
from metalvla_qat.mac.distribution import SAMPLING_VERSION


def run(model, output, episodes=1280, batch_size=8, num_envs=4, resume=False):
    model, output = Path(model).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    with exclusive_lock(output / ".running"):
        return run_locked(model, output, episodes, batch_size, num_envs)


def run_locked(model, output, episodes, batch_size, num_envs):
    identity = bind_identity(output / "identity.json", dict(schema=2, reward_version=REWARD_VERSION, sampling_version=SAMPLING_VERSION, model=model_identity(model),
                  source=str(model), episodes=episodes, batch_size=batch_size, num_envs=num_envs,
                  evaluation_seed=21270001),
                  occupied=any((output / n).exists() for n in ("rl", "evaluation", "result.json")))
    status_path, log_path = output / "status.json", output / "run.log"
    source_hash = file_hash(model / "adapter.safetensors")
    source = json.loads((model / "config.json").read_text())
    started = time.time()
    stage = "starting"

    def status(value, **fields):
        write_json(status_path, dict(stage=value, updated_unix=time.time(), episodes=episodes,
                   source_model=str(model), output=str(output), **fields))

    def command(value, arguments, attempts=1):
        nonlocal stage
        stage = value
        for attempt in range(attempts):
            status(stage, attempt=attempt+1, command=arguments)
            with log_path.open("a") as stream:
                stream.write(json.dumps(dict(stage=stage, attempt=attempt+1, command=arguments))+"\n")
                stream.flush()
                result = subprocess.run(arguments, cwd=ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            if result.returncode == 0:
                return
            if attempt+1 < attempts:
                time.sleep(5)
        raise subprocess.CalledProcessError(result.returncode, arguments)

    try:
        status(stage, source_sha256=source_hash, batch_size=batch_size, num_envs=num_envs)
        rl = output / "rl"
        receipt_path = output / "rl_complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt != dict(model=model_identity(rl / "model"), result_sha256=file_hash(rl / "result.json")):
                raise ValueError({"rl_output_changed": str(rl)})
        else:
            command("rl_training", [sys.executable, "-u", str(ROOT / "scripts/mac_vla.py"),
                "rl", "--model", str(model), "--output", str(rl), "--episodes", str(episodes),
                "--batch-size", str(batch_size), "--num-envs", str(num_envs)], attempts=3)
        trained = json.loads((rl / "result.json").read_text())
        if trained.get("complete") is not True or trained.get("episodes") != episodes:
            raise ValueError({"rl_result": trained})
        weights = rl / "model"
        metadata = json.loads((weights / "config.json").read_text())
        if metadata["config"]["backbone_sha256"] != source["config"]["backbone_sha256"]:
            raise ValueError({"backbone_sha256": metadata["config"]["backbone_sha256"]})
        write_json(receipt_path, dict(model=model_identity(weights), result_sha256=file_hash(rl / "result.json")))
        launcher = Path(sys.executable).with_name("mjpython")
        if not launcher.exists():
            launcher = Path(sys.base_prefix) / "bin/mjpython"
        evaluation = output / "evaluation"
        evaluation_exists = evaluation.exists()
        command("visual_evaluation", [sys.executable, str(launcher), "-u",
            str(ROOT / "scripts/quick_eval_unfrozen.py"), "--model", str(weights),
            "--output-dir", str(evaluation), "--seed", "21270001", "--device", "mps",
            "--close-after", "10"] + (["--resume"] if evaluation_exists else []))
        evaluated = json.loads((evaluation / "result.json").read_text())
        if evaluated.get("complete") is not True or evaluated.get("episodes") != 10:
            raise ValueError({"evaluation_result": evaluated})
        if model_identity(model) != identity["model"]:
            raise ValueError({"source_model_changed": str(model)})
        report = dict(complete=True, rl_episodes=episodes, actual_updates=trained["actual_updates"],
            evaluation_episodes=evaluated["episodes"], evaluation_successes=evaluated["successes"],
            model=str(weights), source_model=str(model), source_sha256=source_hash,
            backbone_sha256=metadata["config"]["backbone_sha256"],
            source_preserved=True, wall_seconds=time.time()-started)
        write_json(output / "result.json", report)
        status("completed", result=str(output / "result.json"),
               successes=evaluated["successes"], actual_updates=trained["actual_updates"])
        return report
    except Exception as error:
        status("failed", failed_stage=stage, error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes < 2 or args.batch_size < 2 or args.num_envs < 1 or args.episodes % args.batch_size:
        raise ValueError(vars(args))
    run(args.model, args.output, args.episodes, args.batch_size, args.num_envs, args.resume)


if __name__ == "__main__":
    main()

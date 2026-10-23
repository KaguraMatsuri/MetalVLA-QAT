"""One-shot user launchd jobs with bounded recovery and explicit acceptance gates."""

import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time

from .data import file_hash, write_json
from .state import bind_identity, exclusive_lock, model_identity, preflight_rl_budget
from ..score import REWARD_VERSION
from .distribution import SAMPLING_VERSION

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "scripts/mac_vla.py"


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / ".running"):
        return _run(config, directory)


def bind_run(config, directory):
    preflight_rl_budget(directory / "rl", config.get("rl_episodes", 128), 8)
    identity = dict(schema=2, reward_version=REWARD_VERSION, sampling_version=SAMPLING_VERSION, config=config,
                    source_sha256=file_hash(config["source_checkpoint"]),
                    manifest_sha256=file_hash(Path(config["dataset"]) / "raw/manifest.jsonl"))
    return bind_identity(directory / "identity.json", identity,
                  occupied=any((directory / name).exists() for name in ("initial", "bc", "rl", "result.json")))


def _run(config, directory):
    identity = bind_run(config, directory)
    status_path = directory / "status.json"
    log_path = directory / "run.log"

    def status(stage, **values):
        write_json(status_path, dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **values))

    def command(stage, arguments, marker=None, attempts=1, inputs=None, artifacts=()):
        arguments = list(map(str, arguments))
        stage_identity = dict(run=identity, arguments=arguments, inputs=inputs)
        receipt = directory / "stages" / (stage + ".json")
        if marker and receipt.exists():
            prior = json.loads(receipt.read_text())
            current = {str(p.relative_to(directory)): file_hash(p) for p in (marker, *artifacts)}
            if prior != dict(identity=stage_identity, outputs=current):
                raise ValueError({"stage_changed": stage})
            return
        status(stage)
        for attempt in range(attempts):
            with log_path.open("a") as stream:
                stream.write(json.dumps(dict(stage=stage, attempt=attempt+1, arguments=arguments))+"\n")
                stream.flush()
                result = subprocess.run([sys.executable, str(ENTRY), *arguments],
                                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                if marker:
                    outputs = {str(p.relative_to(directory)): file_hash(p) for p in (marker, *artifacts)}
                    write_json(receipt, dict(identity=stage_identity, outputs=outputs))
                return
            if attempt+1 < attempts:
                time.sleep(5)
        raise RuntimeError(f"Stage {stage} failed; inspect {log_path}")

    try:
        command("checks", ["check", "--simulation", "--output", directory / "checks"])
        source = Path(config["source_checkpoint"])
        initial = directory / "initial"
        if not (initial / "config.json").exists():
            command("import_model", ["import", "--checkpoint", source, "--output", initial])
        else:
            imported = json.loads((initial / "config.json").read_text())
            if imported["metadata"].get("source_sha256") != file_hash(source):
                raise ValueError("Initial import source changed; choose a new output directory")
        cache = directory / "features"
        command("cache", ["cache", "--model", initial, "--dataset", config["dataset"],
                              "--output", cache, "--batch-size", 8], attempts=3)
        bc = directory / "bc"
        command("bc", ["bc", "--model", initial, "--cache", cache, "--output", bc,
                       "--steps", config.get("bc_steps", 12000), "--batch-size", 32], bc / "result.json", attempts=3,
                inputs=dict(model=model_identity(initial), cache=file_hash(cache / "complete.json")),
                artifacts=(bc / "best_master/config.json", bc / "best_master/adapter.safetensors", bc / "model/config.json", bc / "model/adapter.safetensors"))
        rl = directory / "rl"
        command("rl", ["rl", "--model", bc / "best_master", "--output", rl,
                       "--episodes", config.get("rl_episodes", 128), "--num-envs", 4],
                rl / "result.json", attempts=3, inputs=model_identity(bc / "best_master"),
                artifacts=(rl / "model/config.json", rl / "model/adapter.safetensors"))
        runtime = directory / "performance"
        command("benchmark", ["benchmark", "--model", rl / "model", "--dataset", config["dataset"], "--output", runtime],
                runtime / "result.json", inputs=model_identity(rl / "model"))
        reference, optimized = directory / "reference50", directory / "optimized50"
        command("evaluate_reference", ["evaluate", "--model", rl / "model", "--output", reference,
                                      "--episodes", 50, "--num-envs", 4], reference / "result.json", attempts=3, inputs=model_identity(rl / "model"))
        command("evaluate_optimized", ["evaluate", "--model", rl / "model", "--output", optimized,
                                      "--episodes", 50, "--num-envs", 4, "--profile", runtime / "result.json"],
                optimized / "result.json", attempts=3,
                inputs=dict(model=model_identity(rl / "model"), profile=file_hash(runtime / "result.json")))
        before = json.loads((reference / "result.json").read_text())
        after = json.loads((optimized / "result.json").read_text())
        performance = json.loads((runtime / "result.json").read_text())
        mean_before = sum(r["placement_error_m"] for r in before["results"])/50
        mean_after = sum(r["placement_error_m"] for r in after["results"])/50
        behavior_gate = after["successes"] >= before["successes"] and mean_after <= mean_before*1.10
        allowed = behavior_gate and performance["latency_target_passed"] and performance["throughput_target_passed"]
        report = dict(complete=True, reference_successes=before["successes"], optimized_successes=after["successes"],
                      placement_error_before_m=mean_before, placement_error_after_m=mean_after,
                      closed_loop_gate_passed=behavior_gate, default_switch_allowed=allowed,
                      performance=performance, source_checkpoint=str(source), cache=str(cache),
                      task_success_claimed=after["successes"] > 0, task_episodes=50,
                      task_all_success=after["successes"] == 50, acceptance_scope="runtime_migration")
        write_json(directory / "result.json", report)
        if allowed:
            write_json(directory / "accepted_runtime.json", dict(model=str(rl / "model"),
                        profile=performance["selected"], source_report=str(directory / "result.json")))
        status("completed" if allowed else "completed_targets_not_met", result=str(directory / "result.json"),
               default_switch_allowed=allowed)
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


def launch(config_path):
    if sys.platform != "darwin":
        raise RuntimeError("launchd requires macOS")
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    directory = Path(config["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    bind_run(config, directory)
    label = "org.metalvla.run."+hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    domain = f"gui/{os.getuid()}"
    service = domain+"/"+label
    plist = directory / "job.plist"
    job = dict(Label=label, ProgramArguments=["/usr/bin/caffeinate", "-i", sys.executable, str(ENTRY),
                "run", "--config", str(config_path)], WorkingDirectory=str(ROOT),
               RunAtLoad=True, KeepAlive=False, ProcessType="Background",
               StandardOutPath=str(directory / "launchd.stdout.log"),
               StandardErrorPath=str(directory / "launchd.stderr.log"),
               EnvironmentVariables={"PYTHONUNBUFFERED": "1"})
    with plist.open("wb") as stream:
        plistlib.dump(job, stream)
    current = subprocess.run(["launchctl", "print", service], capture_output=True, text=True)
    if current.returncode:
        subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    elif not (directory / "result.json").exists():
        subprocess.run(["launchctl", "kickstart", service], check=True)
    result = subprocess.run(["launchctl", "print", service], capture_output=True, text=True, check=True)
    write_json(directory / "launchd.json", dict(label=label, service=service, plist=str(plist),
                                               interpreter=sys.executable, status=result.stdout))
    print(json.dumps(dict(service=service, output=str(directory))), flush=True)
    return service

"""Process locks and identities for recoverable local jobs."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from .data import file_hash, write_json

GPU_LOCK = Path.home() / ".cache/metalvla/gpu.lock"


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the inode after unlocking: unlinking permits two independent locks.
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError({"active_job_lock": str(path)}) from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()))
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def bind_identity(path, identity, *, occupied=False):
    path = Path(path)
    identity = json.loads(json.dumps(identity))
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError({"identity_changed": str(path)})
    elif occupied:
        raise ValueError({"missing_identity": str(path)})
    else:
        write_json(path, identity)
    return identity


def model_identity(directory):
    directory = Path(directory)
    names = ["config.json", "adapter.safetensors"]
    payload = json.loads((directory / "config.json").read_text())
    if payload["config"].get("backbone_sha256"):
        names.append("qwen_mlp.safetensors")
    return {name: file_hash(directory / name) for name in names}


def resume_exists(directory):
    directory = Path(directory)
    if (directory / "current.json").exists():
        return True
    if (directory / "progress.json").exists():
        raise ValueError({"uncommitted_legacy_resume": str(directory)})
    return False


def resume_directory(directory):
    directory = Path(directory)
    pointer = json.loads((directory / "current.json").read_text())
    name = pointer["generation"]
    if len(name) != 32 or any(c not in "0123456789abcdef" for c in name):
        raise ValueError({"invalid_generation": name})
    generation = directory / "generations" / name
    if file_hash(generation / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError({"resume_manifest_changed": name})
    manifest = json.loads((generation / "manifest.json").read_text())
    files = {str(p.relative_to(generation)): p for p in generation.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if set(files) != set(manifest) or any(file_hash(files[n]) != digest for n, digest in manifest.items()):
        raise ValueError({"resume_files_changed": name})
    return generation


def validate_rl_budget(episodes, batch_size, completed=0):
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise ValueError({"rl_completed_episodes": completed})
    if completed > episodes:
        raise ValueError("RL episode budget is below the saved progress")
    if episodes < 2 or batch_size < 2 or (episodes-completed) % batch_size == 1:
        raise ValueError("RL final batch must have at least two episodes")


def preflight_rl_budget(directory, episodes, batch_size):
    """Read committed progress without importing a model or initializing Metal."""
    completed = 0
    if resume_exists(directory):
        generation = resume_directory(directory)
        completed = json.loads((generation / "progress.json").read_text())["episodes"]
    validate_rl_budget(episodes, batch_size, completed)
    return completed

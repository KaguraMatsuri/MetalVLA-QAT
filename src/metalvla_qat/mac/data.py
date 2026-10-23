"""Bounded CPU prefetch, signed feature caches and episode-local histories."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def episode_path(directory, row):
    name = hashlib.sha256(row["file"].encode()).hexdigest()[:20] + ".npz"
    return Path(directory) / row["split"] / name


def read_manifest(root):
    path = Path(root) / "raw/manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("schema") != "ur5e_pick_place_pose_v1" or not row.get("success")
                       or row.get("split") not in ("train", "val") for row in rows):
        raise ValueError("Unexpected expert manifest contract")
    return rows, file_hash(path)


def load_episode(root, row, config):
    path = Path(root) / row["file"]
    if file_hash(path) != row["sha256"]:
        raise ValueError(f"Source episode changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        desired = data["desired"].astype(np.float32)
        if len(desired) == 0 or data["state"].shape != (len(desired), 8) or data["previous"].shape != (len(desired), 8):
            raise ValueError("Misaligned expert observation/control arrays")
        anchors = np.arange(0, len(desired), config.execute_steps)
        future = anchors[:, None] + np.arange(config.horizon)[None]
        valid = future < len(desired)
        labels = desired[np.minimum(future, len(desired)-1)]
        lo, hi = np.asarray(config.workspace_low), np.asarray(config.workspace_high)
        xyz = (labels[..., :3]-(lo+hi)/2)/((hi-lo)/2)
        target = np.concatenate((xyz, np.sin(labels[..., 3:4]), np.cos(labels[..., 3:4]),
                                 labels[..., 4:5]*2-1), axis=-1).astype(np.float32)
        result = dict(raw=data["images"][anchors].copy(), state=data["state"][anchors].copy(),
                      previous=data["previous"][anchors].copy(), target=target, valid=valid,
                      anchors=anchors, region=np.array(row["region"]), ordinal=np.array(row["ordinal"]))
    return result


def build_cache(root, directory, encoder, config, batch_size=8):
    rows, manifest_hash = read_manifest(root)
    directory = Path(directory)
    signature = dict(encoder=encoder.signature(), manifest_sha256=manifest_hash,
                     policy=asdict(config), dataset_root=str(Path(root).resolve()), validated=True)
    signature = json.loads(json.dumps(signature))
    signature_path = directory / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Feature cache signature changed; use a new cache directory")
    write_json(signature_path, signature)
    completed_path = directory / "complete.json"
    if completed_path.exists():
        validate_cache_files(directory, rows)
        write_json(directory / "progress.json", dict(complete=True, episodes=len(rows), target=len(rows)))
        return signature
    pending_rows = [r for r in rows if not episode_path(directory, r).exists()]
    completed = len(rows)-len(pending_rows)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for group_start in range(0, len(pending_rows), 4):
            group = pending_rows[group_start:group_start+4]
            episodes = [load_episode(root, row, config) for row in group]
            flat = [(e, i) for e, episode in enumerate(episodes) for i in range(len(episode["anchors"]))]
            features = [np.empty((len(e["anchors"]), encoder.hidden_size), np.float32) for e in episodes]
            batches = [flat[i:i+batch_size] for i in range(0, len(flat), batch_size)]
            queued = {}

            def prepare(index):
                raw = np.stack([episodes[e]["raw"][i] for e, i in batches[index]])
                return encoder.prepare(raw, "pick blue cube and place on red target")

            for index in range(min(2, len(batches))):
                queued[index] = pool.submit(prepare, index)
            for index, batch in enumerate(batches):
                encoded = encoder.encode_prepared(queued.pop(index).result())
                if index + 2 < len(batches):
                    queued[index+2] = pool.submit(prepare, index+2)
                for value, (episode_index, anchor) in zip(encoded, batch, strict=True):
                    features[episode_index][anchor] = value
            for row, episode, feature in zip(group, episodes, features, strict=True):
                episode.pop("raw")
                episode["features"] = feature
                write_npz(episode_path(directory, row), episode)
                completed += 1
            write_json(directory / "progress.json", dict(complete=completed==len(rows),
                                                         episodes=completed, target=len(rows)))
            print(json.dumps(dict(stage="cache", episodes=completed, total=len(rows))), flush=True)
    hashes = {}
    for row in rows:
        path = episode_path(directory, row)
        read_cached_episode(path, config)
        hashes[str(path.relative_to(directory))] = file_hash(path)
    write_json(completed_path, dict(signature_sha256=file_hash(signature_path), files=hashes))
    write_json(directory / "progress.json", dict(complete=True, episodes=completed, target=len(rows)))
    return signature


def validate_cache_files(directory, rows, split=None):
    directory = Path(directory)
    completed = json.loads((directory / "complete.json").read_text())
    paths = {str(episode_path(directory, r).relative_to(directory)): episode_path(directory, r) for r in rows}
    selected = {str(episode_path(directory, r).relative_to(directory)) for r in rows if split is None or r["split"] == split}
    if (completed["signature_sha256"] != file_hash(directory / "signature.json")
            or set(completed["files"]) != set(paths)
            or any(not p.exists() or (n in selected and file_hash(p) != completed["files"][n]) for n, p in paths.items())):
        raise ValueError({"cache_files_changed": str(directory)})


def read_cached_episode(path, config):
    with np.load(path, allow_pickle=False) as data:
        episode = {name: data[name].copy() for name in ("features", "state", "previous", "target", "valid", "anchors")}
    count = len(episode["anchors"])
    shapes = dict(features=(count, config.feature_dim), state=(count, 8), previous=(count, 8),
                  target=(count, config.horizon, 6), valid=(count, config.horizon))
    if count == 0 or any(episode[name].shape != shape for name, shape in shapes.items()):
        raise ValueError({"cache_shape": str(path)})
    if any(not np.isfinite(episode[name]).all() for name in ("features", "state", "previous", "target")):
        raise ValueError("Nonfinite cached data")
    if episode["valid"].dtype != np.bool_:
        raise ValueError("Cached action mask changed")
    return episode


def load_cache(directory, split, config, require_validated=True):
    directory = Path(directory)
    signature = json.loads((directory / "signature.json").read_text())
    if require_validated and not signature.get("validated"):
        raise ValueError("Imported feature cache requires real-RGB verification")
    expected = signature["policy"]
    if expected != json.loads(json.dumps(asdict(config))):
        raise ValueError("Policy/base differs from feature cache")
    rows, manifest_hash = read_manifest(signature["dataset_root"])
    if manifest_hash != signature["manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    validate_cache_files(directory, rows, split)
    episodes = []
    for row in rows:
        if row["split"] != split:
            continue
        path = episode_path(directory, row)
        episode = read_cached_episode(path, config)
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"No {split} cache episodes")
    return episodes, signature


def sample(episode, anchor, config):
    n = config.history_length
    indices = np.arange(anchor-n, anchor)
    valid = indices >= 0
    indices = np.maximum(indices, 0)
    history = np.concatenate((episode["features"][indices], episode["state"][indices],
                              episode["previous"][indices]), axis=-1)
    return (episode["features"][anchor], episode["state"][anchor], episode["previous"][anchor],
            history, valid, episode["target"][anchor], episode["valid"][anchor])


def stack_samples(samples):
    import mlx.core as mx
    return tuple(mx.array(np.stack([sample[i] for sample in samples])) for i in range(7))


def random_batch(episodes, rng, batch_size, config):
    samples = []
    for _ in range(batch_size):
        episode = episodes[int(rng.integers(len(episodes)))]
        samples.append(sample(episode, int(rng.integers(len(episode["features"]))), config))
    return stack_samples(samples)


def validation_batches(episodes, config, batch_size=64):
    pending = []
    for episode in episodes:
        for anchor in range(len(episode["features"])):
            pending.append(sample(episode, anchor, config))
            if len(pending) == batch_size:
                yield stack_samples(pending)
                pending = []
    if pending:
        yield stack_samples(pending)

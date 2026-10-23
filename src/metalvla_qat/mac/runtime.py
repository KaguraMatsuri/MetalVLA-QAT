"""Spawned MuJoCo environments; one central Qwen/MPS and MLX policy."""

import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import sys
import time
import traceback

import mlx.core as mx
import numpy as np

from .distribution import SAMPLING_VERSION, gripper_logits
from .data import write_json, write_npz
from .encoder import assert_base_matches

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


class VisiblePolicy:
    """Expose the MLX policy to the existing visible UR execution controller."""
    def __init__(self, directory, device="mps"):
        import torch
        from .model import load_model
        from .cli import encoder_for
        from ..qwen_pose_vla import QwenPoseVLA
        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        self.policy, payload = load_model(directory)
        self.encoder = encoder_for(self.policy, device=device)
        self.config = self.policy.config
        self.vocabulary = payload["metadata"]["vocabulary"]
        self.words = {value: word for word, value in self.vocabulary.items()}
        self.qat_prepared = False
        device = next(self.encoder.model.parameters()).device
        self.position_origin = torch.tensor([(a+b)/2 for a,b in zip(self.config.workspace_low, self.config.workspace_high)], device=device)
        self.position_scale = torch.tensor([(b-a)/2 for a,b in zip(self.config.workspace_low, self.config.workspace_high)], device=device)
        self.decode = QwenPoseVLA.decode.__get__(self)
        self.execution_prefix = QwenPoseVLA.execution_prefix.__get__(self)

    def parameters(self):
        return self.encoder.model.parameters()

    def empty_history(self, batch, device=None):
        import torch
        device = device or next(self.parameters()).device
        return (torch.zeros(batch, 0, self.config.feature_dim+16, device=device),
                torch.zeros(batch, 0, dtype=torch.bool, device=device))

    def encode_instruction(self, tokens):
        return tokens

    def __call__(self, images, tokens, state, previous, history, valid):
        import torch
        pixels = images.detach().cpu().permute(0, 1, 3, 4, 2).mul(255).round().to(torch.uint8).numpy()
        instructions = [" ".join(self.words[token] for token in row if token)
                        for row in tokens.cpu().tolist()]
        features = self.encoder.encode(pixels, instructions)
        actions = self.policy.actions(mx.array(features), mx.array(state.cpu().numpy()),
                                     mx.array(previous.cpu().numpy()))
        mx.eval(actions)
        return (torch.from_numpy(np.asarray(actions).copy()).to(state.device),
                state.new_zeros((len(state), 3)), history, valid)

    def close(self):
        self.encoder.close()


def _environment(pipe, name, shape):
    """No Qwen or MLX model is loaded in an environment process."""
    sys.path.insert(0, str(SCRIPTS))
    from ur_pick_place import URTask
    import ur_pose_worker as controls
    memory = shared_memory.SharedMemory(name=name)
    images = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
    task = None
    try:
        task = URTask(width=shape[2], height=shape[1], render=True)
        counters = dict(closed_ticks=0, empty_ticks=0, alignment=None, recovery_steps=0)
        while True:
            command, payload = pipe.recv()
            if command == "close":
                break
            if command == "reset":
                task.reset(payload["seed"], payload["region"], payload["ordinal"])
                counters = dict(closed_ticks=0, empty_ticks=0, alignment=None, recovery_steps=0)
            if command in ("reset", "observe"):
                observation = task.observe()
                images[:] = observation["images"]
                pipe.send(dict(state=observation["state"], previous=observation["previous"]))
            elif command == "execute":
                missed, executed = False, 0
                for index, action in enumerate(payload["actions"]):
                    task.execute(action[:3], float(action[3]), bool(action[4]))
                    executed += 1
                    measured = float(task.state()[-1])
                    missed, counters["closed_ticks"], counters["empty_ticks"] = controls.empty_grasp(
                        bool(action[4]), measured, counters["closed_ticks"], counters["empty_ticks"])
                    task.trace[-1].update(decision=payload["decision"], prefix_step=index,
                                         action_source="policy", empty_grasp_detected=missed,
                                         measured_gripper=measured)
                    if bool(action[4]):
                        error = float(np.linalg.norm(task.data.geom_xpos[task.pad_ids].mean(0)-task.cube_position()))
                        counters["alignment"] = error if counters["alignment"] is None else min(error, counters["alignment"])
                    if missed or task.success:
                        break
                pipe.send(dict(executed_steps=executed, missed=missed, success=bool(task.success)))
            elif command == "recover":
                controls.retry_open_raise(task)
                counters["closed_ticks"] = counters["empty_ticks"] = 0
                counters["recovery_steps"] += 6
                pipe.send(dict(recovered=True))
            elif command == "result":
                result = controls.precision_score(task, payload["decisions"], payload["retries"], counters["alignment"])
                failure = ("invalid_policy_action" if payload.get("policy_error") else controls.failure_type(result, task.trace))
                result.update(failure_type=failure,
                              grasp_retries=payload["retries"], recovery_steps=counters["recovery_steps"],
                              expert_takeover=False)
                if payload.get("policy_error"):
                    result.update(success=False, score=0., policy_error=payload["policy_error"])
                pipe.send(dict(result=result, trace=task.trace))
    except BaseException as error:
        try:
            pipe.send(dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if task is not None:
            task.close()
        memory.close()
        pipe.close()


class Environments:
    def __init__(self, count=4, width=320, height=180):
        context = mp.get_context("spawn")
        self.lanes = []
        try:
            for _ in range(count):
                shape = (3, height, width, 3)
                memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
                pipe, child = context.Pipe()
                process = context.Process(target=_environment, args=(child, memory.name, shape))
                process.start()
                child.close()
                self.lanes.append(dict(pipe=pipe, process=process, memory=memory,
                                       images=np.ndarray(shape, np.uint8, buffer=memory.buf)))
        except BaseException:
            self.close()
            raise

    def send(self, index, command, payload=None):
        self.lanes[index]["pipe"].send((command, payload or {}))

    def receive(self, index, timeout=180):
        lane = self.lanes[index]
        if not lane["pipe"].poll(timeout):
            raise TimeoutError(f"MuJoCo environment {index} did not respond")
        result = lane["pipe"].recv()
        if "error" in result:
            raise RuntimeError(result["error"] + "\n" + result["traceback"])
        return result

    def close(self):
        for lane in self.lanes:
            try:
                if lane["process"].is_alive():
                    try:
                        lane["pipe"].send(("close", {}))
                    except (OSError, EOFError):
                        pass
                lane["process"].join(timeout=10)
                if lane["process"].is_alive():
                    lane["process"].terminate()
                    lane["process"].join(timeout=5)
            finally:
                lane["pipe"].close()
                lane["memory"].close()
                lane["memory"].unlink()


def sample_actions(reference, rng, noise_scale=.5):
    xyz = np.arctanh(np.clip(reference[..., :3], -.999, .999))
    yaw = np.arctan2(reference[..., 3:4], reference[..., 4:5])
    mean = np.concatenate((xyz, yaw), axis=-1)
    latent = rng.normal(mean, np.asarray([.10, .10, .06, .08])*noise_scale).astype(np.float32)
    logits = gripper_logits(reference[..., 5], noise_scale, np)
    probability = 1/(1+np.exp(-logits))
    closed = (rng.random(probability.shape) < probability).astype(np.float32)
    sampled = np.concatenate((np.tanh(latent[..., :3]), np.sin(latent[..., 3:4]),
                               np.cos(latent[..., 3:4]), (2*closed-1)[..., None]), axis=-1)
    return sampled, latent, closed


def rollout_wave(environments, policy, encoder, seeds, ordinals, rng=None, noise_scale=.5,
                 max_decisions=15, retries=2):
    assert_base_matches(policy, encoder)
    count = len(seeds)
    if count > len(environments.lanes):
        raise ValueError("Too many active environments")
    lanes = []
    for i, (seed, ordinal) in enumerate(zip(seeds, ordinals, strict=True)):
        environments.send(i, "reset", dict(seed=int(seed), region=int(ordinal)%9, ordinal=int(ordinal)))
        history, valid = policy.empty_history(1)
        lanes.append(dict(done=False, retries=0, decisions=[], history=history, valid=valid,
                          seed=int(seed), ordinal=int(ordinal), started=time.monotonic()))
    observations = {i: environments.receive(i) for i in range(count)}
    packed = policy.snapshot()
    complete_task_block = policy.config.history_length == 0 and policy.config.execute_steps == policy.config.horizon
    budget = min(max_decisions, retries+1) if complete_task_block else max_decisions
    for decision in range(budget):
        active = [i for i in range(count) if not lanes[i]["done"]]
        if not active:
            break
        if decision:
            for i in active:
                environments.send(i, "observe")
            observations = {i: environments.receive(i) for i in active}
        raw = np.stack([environments.lanes[i]["images"] for i in active])
        features = encoder.encode(raw, "pick blue cube and place on red target")
        state = np.stack([observations[i]["state"] for i in active])
        previous = np.stack([observations[i]["previous"] for i in active])
        history = mx.concatenate([lanes[i]["history"] for i in active], axis=0)
        valid = mx.concatenate([lanes[i]["valid"] for i in active], axis=0)
        prediction, next_history, next_valid = packed(mx.array(features), mx.array(state), mx.array(previous), history, valid)
        mx.eval(prediction, next_history, next_valid)
        reference = np.asarray(prediction)
        if rng is None:
            actions, latent, closed = reference, None, None
        else:
            actions, latent, closed = sample_actions(reference, rng, noise_scale)
        executed_lanes = []
        for j, i in enumerate(active):
            record = dict(features=features[j:j+1].copy(), state=state[j:j+1].copy(),
                          previous=previous[j:j+1].copy(), history=np.asarray(history[j:j+1]).copy(),
                          valid=np.asarray(valid[j:j+1]).copy(), reference=reference[j:j+1].copy())
            if rng is not None:
                record.update(latent=latent[j:j+1].copy(), closed=closed[j:j+1].copy())
            lanes[i]["decisions"].append(record)
            lanes[i]["history"], lanes[i]["valid"] = next_history[j:j+1], next_valid[j:j+1]
            try:
                xyz, yaw, jaw = policy.decode(actions[j:j+1])
            except ValueError as error:
                record["executed_steps"] = 0
                lanes[i].update(done=True, policy_error=str(error))
                continue
            chunk = np.concatenate((xyz[0, :policy.config.execute_steps],
                                    yaw[0, :policy.config.execute_steps, None],
                                    jaw[0, :policy.config.execute_steps, None]), axis=-1)
            environments.send(i, "execute", dict(actions=chunk, decision=decision+1))
            executed_lanes.append(i)
        for i in executed_lanes:
            outcome = environments.receive(i)
            lanes[i]["decisions"][-1]["executed_steps"] = outcome["executed_steps"]
            if outcome["success"]:
                lanes[i]["done"] = True
            elif outcome["missed"]:
                if lanes[i]["retries"] < retries and decision+1 < budget:
                    lanes[i]["retries"] += 1
                    environments.send(i, "recover")
                    environments.receive(i)
                else:
                    lanes[i]["done"] = True
            elif complete_task_block:
                lanes[i]["done"] = True
    results = []
    for i, lane in enumerate(lanes):
        environments.send(i, "result", dict(decisions=len(lane["decisions"]), retries=lane["retries"],
                                           policy_error=lane.get("policy_error")))
        result = environments.receive(i)
        result["result"].update(seed=lane["seed"], ordinal=lane["ordinal"], region=lane["ordinal"]%9,
                                wall_seconds=time.monotonic()-lane["started"], viewer_opened=False,
                                sampling_version=SAMPLING_VERSION)
        result["decisions"] = lane["decisions"]
        results.append(result)
    return results


def save_experiences(path, experiences):
    arrays, description = {}, []
    for i, episode in enumerate(experiences):
        decisions = []
        for j, decision in enumerate(episode["decisions"]):
            row = dict(executed_steps=decision["executed_steps"])
            for key, value in decision.items():
                if key != "executed_steps":
                    name = f"e{i}_d{j}_{key}"
                    arrays[name] = value
                    row[key] = name
            decisions.append(row)
        description.append(dict(result=episode["result"], decisions=decisions))
    arrays["metadata"] = np.array(json.dumps(description))
    write_npz(path, arrays)


def load_experiences(path):
    with np.load(path, allow_pickle=False) as data:
        description = json.loads(str(data["metadata"]))
        for episode in description:
            for row in episode["decisions"]:
                for key in list(row):
                    if key != "executed_steps":
                        row[key] = data[row[key]].copy()
        return description

"""Shared expert collection and UR execution for the single Qwen/MLX route."""
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from metalvla_qat.mac.data import write_json as save_json
from metalvla_qat.score import apply_reward
from ur_pick_place import (URTask, SCHEMA, INSTRUCTION, CAMERAS, PERIOD,
    WORKSPACE_LOW, WORKSPACE_HIGH, episode_metadata, file_sha256)
INVALID_YAW_ERROR = "Undefined yaw: sin/cos vector is zero"

def torch_modules():
    import torch
    from metalvla_qat.rgb import preprocess_rgb
    return torch, None, None, preprocess_rgb, None, None

def budget(args):
    if (args.run / "STOP").exists():
        raise InterruptedError("STOP requested")
    if shutil.disk_usage(args.run).free < args.min_free_gb*1024**3:
        raise InterruptedError("disk reserve reached")

def failure_type(result,trace):
    if result["success"]:
        return "success"
    if not result["bilateral_grasp"]:
        closest=min(np.linalg.norm(np.asarray(row["tcp"])-row["cube"]) for row in trace)
        return "alignment" if closest>.04 else "closure"
    if not result["lifted"]:
        return "lift"
    if result["placement_error_m"]>=.02:
        return "transfer"
    return "release_or_stability"

def manifest(args):
    path = args.run / "raw/manifest.jsonl"
    if not path.exists():
        return []
    rows = []
    lines=path.read_text().splitlines()
    for index,line in enumerate(lines):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index!=len(lines)-1:
                    raise
                # Only an incomplete final append is recoverable from the NPZ.
                print("Recovering interrupted final manifest append",flush=True)
    if len({(r["region"],r["ordinal"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate region/ordinal in manifest")
    return rows

def write_npz(path, arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def clean_episode(values, task):
    arrays = {key:np.asarray(value) for key,value in values.items()}
    frames = len(arrays["desired"])
    if frames < 8 or arrays["images"].shape != (frames,3,task.height,task.width,3):
        raise ValueError("Images and desired actions are misaligned")
    if arrays["images"].dtype != np.uint8 or not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Invalid numeric/image data")
    if arrays["state"].shape != (frames,8) or arrays["previous"].shape != (frames,8):
        raise ValueError("Invalid UR padded state/control contract")
    if arrays["desired"].shape != (frames,5) or arrays["offset"].shape != (frames,3):
        raise ValueError("Invalid pose supervision")
    if not np.all(np.isin(arrays["desired"][:,4], [0,1])) or np.abs(arrays["previous"]).max()>1.00001:
        raise ValueError("Invalid gripper/control range")
    if not np.all(arrays["state"][:,6] == 0) or not np.all(arrays["previous"][:,6] == 0):
        raise ValueError("UR zero-padding slot changed")
    # Uniform timestamps and previous control alignment, including the reset.
    arrays["time"] = np.arange(frames,dtype=np.float64)*PERIOD
    applied = np.array([row["executed_control"] for row in task.trace],dtype=np.float32)
    if len(applied) != frames or not np.allclose(arrays["previous"][1:],applied[:-1],atol=1e-6):
        raise ValueError("Past executed controls are shifted relative to observations")
    arrays["executed_control"] = applied
    arrays["tcp"] = np.array([row["tcp"] for row in task.trace],dtype=np.float32)
    arrays["cube"] = np.array([row["cube"] for row in task.trace],dtype=np.float32)
    arrays["goal"] = task.goal.astype(np.float32)
    # Check the actual three saved RGB cameras. Requiring every object in the
    # overhead image alone would reject useful wrist observations under occlusion.
    first = arrays["images"][0].astype(np.int16)
    blue = (first[:,:,:,2]>first[:,:,:,0]+35)&(first[:,:,:,2]>first[:,:,:,1]+20)
    red = (first[:,:,:,0]>first[:,:,:,1]+50)&(first[:,:,:,0]>first[:,:,:,2]+50)
    blue_counts,red_counts=blue.sum((1,2)),red.sum((1,2))
    arrays["initial_visibility_pixels"]=np.stack((blue_counts,red_counts),-1)
    if blue_counts.max()<8 or red_counts.max()<8:
        raise ValueError("Cube or red marker is not visible in any recorded RGB camera")
    return arrays

def collect(args):
    raw = args.run/"raw"
    (raw/"successes").mkdir(parents=True,exist_ok=True)
    (raw/"failures").mkdir(exist_ok=True)
    rows = manifest(args)
    # NPZ includes its own metadata. Recover atomically saved episodes whose
    # manifest append was interrupted, rather than silently collecting twice.
    known = {r["file"] for r in rows}
    for path in sorted((raw/"successes").glob("*.npz")):
        relative = str(path.relative_to(args.run))
        if relative not in known:
            with np.load(path,allow_pickle=False) as saved:
                meta = json.loads(str(saved["metadata"]))
            if meta["schema"] != SCHEMA or meta["seed"] != args.seed or not meta["success"]:
                raise ValueError("Incompatible orphaned episode")
            is_val=meta["ordinal"]%2==1 if args.smoke else meta["ordinal"]%10==9
            meta.update(file=relative,sha256=file_sha256(path),split="val" if is_val else "train")
            rows.append(meta)
    # Rewrite atomically also repairs an interrupted manifest append on resume.
    temporary = raw/"manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(r)+"\n" for r in rows))
    temporary.replace(raw/"manifest.jsonl")
    counts = [sum(r["region"]==region for r in rows) for region in range(9)]
    for region in range(9):
        if sorted(r["ordinal"] for r in rows if r["region"]==region)!=list(range(counts[region])):
            raise ValueError("Noncontiguous per-cell ordinals; refusing duplicate or biased resume")
    if any(r["schema"] != SCHEMA or r["seed"] != args.seed for r in rows):
        raise ValueError("Mixed dataset schema/seed; choose a separate run directory")
    task = URTask(args.width,args.height)
    attempt = 0
    started = time.monotonic()
    try:
        while min(counts)<args.quota:
            budget(args)
            region = min(range(9),key=lambda r:counts[r])
            ordinal = counts[region]
            for retry in range(args.max_retries):
                budget(args)
                attempt += 1
                try:
                    task.reset(args.seed,region,ordinal,retry)
                    values,result = task.expert()
                    arrays = clean_episode(values,task) if result["success"] else None
                    if arrays is None:
                        raise RuntimeError("Physical expert failed stable release")
                    meta = episode_metadata(task,result)
                    path = raw/"successes"/f"cell_{region+1}_episode_{ordinal+1:06d}.npz"
                    if path.exists():
                        raise FileExistsError(path)
                    arrays["metadata"] = np.array(json.dumps(meta))
                    write_npz(path,arrays)
                    is_val=ordinal%2==1 if args.smoke else ordinal%10==9
                    meta.update(file=str(path.relative_to(args.run)),sha256=file_sha256(path),
                                split="val" if is_val else "train")
                    rows.append(meta)
                    with (raw/"manifest.jsonl").open("a") as stream:
                        stream.write(json.dumps(meta)+"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    counts[region] += 1
                    save_json(args.run/"collection.json",dict(counts=counts,target_per_cell=args.quota,
                        successful_expert_episodes=sum(counts),attempts_this_stage=attempt,
                        seconds_this_stage=time.monotonic()-started, autonomous_successes=0))
                    print(f"COLLECT cell={region+1} count={counts[region]}/{args.quota} total={sum(counts)} score={result['score']:.3f}",flush=True)
                    break
                except (RuntimeError,ValueError) as error:
                    # Keep failures as compact traces; they are not successful BC labels.
                    save_json(raw/"failures"/f"cell{region}_episode{ordinal}_retry{retry}.json",
                        dict(error=str(error),region=region,ordinal=ordinal,retry=retry,trace=task.trace))
                    print(f"COLLECT failure cell={region+1} retry={retry}: {error}",flush=True)
            else:
                raise RuntimeError(f"Cell {region+1} stratum failed {args.max_retries} times; refusing biased omission")
    finally:
        task.close()
    save_json(args.output/"result.json",dict(counts=counts,episodes=sum(counts),expert_only=True))


def empty_grasp(closed,measured_gripper,closed_ticks,empty_ticks):
    """UR/2F85 encoder-only trigger; cube geometry is not a control input.

    Calibration for this fixed 5-cm cube: contact closure .418-.449;
    empty closure .978. Require .5s closing and two .1s empty samples.
    """
    closed_ticks=closed_ticks+1 if closed else 0
    empty_ticks=empty_ticks+1 if closed and measured_gripper>=.95 else 0
    return closed_ticks>=5 and empty_ticks>=2,closed_ticks,empty_ticks

def retry_open_raise(task):
    """Bounded recovery using measured TCP, without object-dependent targets."""
    target=task.tcp_position().copy()
    target[2]=min(WORKSPACE_HIGH[2],target[2]+.08)
    yaw=float(task.last_target[3])
    for _ in range(6):
        task.execute(target,yaw,False)
        task.trace[-1].update(action_source="retry_open_raise")

def precision_score(task,decisions,retries,alignment_error):
    """Reward-only simulator truth; physical success keeps the original gate."""
    result=task.score(decisions)
    result.update(legacy_score=result["score"], precision_reward=True,
                  grasp_alignment_best_m=alignment_error, grasp_retries=retries)
    apply_reward(result)
    result["retry_penalty"] = result["reward_components"]["reinference_penalty"]
    return result

def rollout(args, task, model, seed, region, ordinal):
    torch, _, _, preprocess_rgb, _, _ = torch_modules()
    device = next(model.parameters()).device
    task.reset(seed, region, ordinal)
    retry_limit = getattr(args, 'grasp_retries', 0)
    precision = getattr(args, 'precision_reward', False)
    complete_block = model.config.execute_steps == model.config.horizon and (not getattr(model.config, 'replan_after_chunk', False))
    retries = 0
    empty_detections = 0
    alignment_error = None
    policy_steps = 0
    recovery_steps = 0
    closed_ticks = empty_ticks = 0
    history, valid = model.empty_history(1, device)
    tokens = torch.arange(1, len(model.vocabulary) + 1, device=device)[None]
    decisions = []
    fusion_inputs = []
    rejected_chunk = None
    hook = None
    started = time.monotonic()
    try:
        count = min(args.max_decisions, retry_limit + 1) if retry_limit and complete_block else args.max_decisions
        for decision in range(count):
            budget(args)
            observation = task.observe()
            images = preprocess_rgb(torch.from_numpy(observation['images'])[None], model.config.image_size).to(device)
            state = torch.from_numpy(observation['state'])[None].to(device)
            previous = torch.from_numpy(observation['previous'])[None].to(device)
            with torch.no_grad():
                actions, _, history, valid = model(images, tokens, state, previous, history, valid)
                prefix = model.execution_prefix(actions)
                try:
                    xyz, yaw, closed = model.decode(prefix)
                except ValueError as error:
                    if str(error) != INVALID_YAW_ERROR:
                        raise
                    rejected_chunk = dict(event='policy_action_chunk_rejected', decision=decision + 1, reason='invalid_yaw', error_type=type(error).__name__, error=str(error), predicted_steps=int(prefix.shape[1]), executed_steps=0, fallback_used=False)
                    task.trace.append(rejected_chunk.copy())
                    break
            missed = False
            executed = 0
            for t in range(model.config.execute_steps):
                task.execute(xyz[0, t].cpu().numpy(), float(yaw[0, t]), bool(closed[0, t]))
                task.trace[-1].update(decision=decision + 1, prefix_step=t)
                executed += 1
                policy_steps += 1
                if retry_limit:
                    measured = float(task.state()[-1])
                    missed, closed_ticks, empty_ticks = empty_grasp(bool(closed[0, t]), measured, closed_ticks, empty_ticks)
                    task.trace[-1].update(action_source='policy', measured_gripper=measured, empty_grasp_detected=missed)
                if precision and bool(closed[0, t]):
                    error = float(np.linalg.norm(task.data.geom_xpos[task.pad_ids].mean(0) - task.cube_position()))
                    alignment_error = error if alignment_error is None else min(alignment_error, error)
                if missed:
                    empty_detections += 1
                    break
                if task.success:
                    break
            if task.success:
                break
            if retry_limit:
                if not missed:
                    if complete_block:
                        break
                    continue
                if retries >= retry_limit or decision + 1 >= count:
                    break
                retries += 1
                retry_open_raise(task)
                recovery_steps += 6
                closed_ticks = empty_ticks = 0
    finally:
        if hook:
            hook.remove()
    if rejected_chunk is not None and not precision:
        result = task.score(decision + 1)
    elif precision:
        result = precision_score(task, decision + 1, retries, alignment_error)
    else:
        result = task.score(decision + 1)
    result['wall_seconds'] = time.monotonic() - started
    result.update(region=region, seed=seed, ordinal=ordinal, expert_takeover=False)
    if rejected_chunk is not None:
        result.update(success=False, score=0.0, failure_type='invalid_policy_action', policy_action_rejected=True, policy_error_type=rejected_chunk['error_type'], policy_error=rejected_chunk['error'], rejected_chunk_steps=rejected_chunk['predicted_steps'], executed_steps_in_rejected_chunk=0)
    else:
        result['failure_type'] = failure_type(result, task.trace)
    if retry_limit:
        result.update(grasp_retries=retries, empty_grasp_detections=empty_detections, policy_steps=policy_steps, recovery_steps=recovery_steps, retry_trigger='measured_gripper_encoder', retry_controller='rule_open_raise', retry_motion_policy='fresh_RGB_state_history_model_inference')
    return (result, decisions)

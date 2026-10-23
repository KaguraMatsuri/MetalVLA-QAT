"""Model import, signed caches, MLX QAT training, evaluation and profiling."""
import argparse
import hashlib
from dataclasses import replace
import json
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from .data import build_cache, file_hash, load_cache, write_json
from .encoder import EncoderConfig, OfficialQwenEncoder
from .model import MacPolicy, PolicyConfig, import_torch_checkpoint, load_model, save_model
from .performance import benchmark, old_jobs_active
from .runtime import Environments, load_experiences, rollout_wave, save_experiences
from .train import reinforce_update, restore_resume, resume_exists, save_resume, train_bc
from .state import GPU_LOCK, bind_identity, exclusive_lock, model_identity, preflight_rl_budget, validate_rl_budget
from ..score import REWARD_VERSION
from .distribution import SAMPLING_VERSION
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / 'data/ur_pick_place_autoloop_v1'

def encoder_for(model, profile=None, device='mps'):
    value = {}
    if profile:
        value = json.loads(Path(profile).read_text())
        value = value.get('selected', value.get('profile', value))
    c = model.config
    if value and value.get('backbone_sha256') != c.backbone_sha256:
        raise ValueError('Runtime profile belongs to another base')
    if not c.backbone_sha256:
        raise ValueError({'backbone_sha256': None})
    dtype = value.get('dtype', getattr(model, '_metadata', {}).get('backbone_dtype', 'float32'))
    config = EncoderConfig(base_model=c.base_model, revision=c.base_revision, image_size=c.image_size, dtype=dtype, attention=value.get('attention', 'eager'), backbone_path=str(model._backbone_file), backbone_sha256=c.backbone_sha256)
    if value.get('base_model', c.base_model) != c.base_model or value.get('revision', c.base_revision) != c.base_revision:
        raise ValueError('Runtime profile belongs to another base')
    return OfficialQwenEncoder(config, device=device)

def evaluate(model, encoder, directory, episodes=50, num_envs=4, seed=20261009):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = directory / 'result.json'
    results = []
    digest = hashlib.sha256()
    mx.eval(model.parameters())
    for name, value in sorted(tree_flatten(model.parameters())):
        digest.update(name.encode())
        digest.update(np.asarray(value).tobytes())
    identity = dict(config=model.config.__dict__, encoder=encoder.signature(), policy_sha256=digest.hexdigest(),
                    scenario=dict(seed=seed, region_modulo=9, max_decisions=15, retries=2, version=1, reward_version=REWARD_VERSION))
    identity = json.loads(json.dumps(identity))
    if existing.exists():
        prior = json.loads(existing.read_text())
        if prior['identity'] != identity:
            raise ValueError('Evaluation base/model config changed')
        results = prior['results']
        if any(r.get('seed') != seed+i or r.get('ordinal') != i or r.get('region') != i % 9 for i, r in enumerate(results)):
            raise ValueError({'evaluation_sequence_changed': str(existing)})
        if len(results) > episodes:
            raise ValueError('Evaluation count shrank')
    if len(results) == episodes:
        return prior
    envs = Environments(num_envs)
    try:
        for start in range(len(results), episodes, num_envs):
            ordinals = list(range(start, min(start + num_envs, episodes)))
            wave = rollout_wave(envs, model, encoder, [seed + i for i in ordinals], ordinals)
            for ordinal, experience in zip(ordinals, wave, strict=True):
                result = experience['result']
                results.append(result)
                write_json(directory / f'episode_{ordinal + 1:03d}.json', dict(result=result, trace=experience['trace']))
                print(json.dumps(dict(stage='evaluate', episode=ordinal + 1, success=result['success'], score=result['score'], retries=result['grasp_retries'])), flush=True)
            write_json(existing, dict(complete=len(results) == episodes, identity=identity, episodes=len(results), successes=sum((r['success'] for r in results)), results=results, viewer_opened=False, expert_takeover=False, qat_mode='mxfp8' if model.fp8 else 'float32'))
    finally:
        envs.close()
    return json.loads(existing.read_text())

def train_rl(model, encoder, directory, episodes=128, batch_size=8, num_envs=4, seed=20261008, learning_rate=1e-06, source_identity=None):
    directory = Path(directory)
    has_resume = resume_exists(directory)
    if not has_resume:
        validate_rl_budget(episodes, batch_size)
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = optim.AdamW(learning_rate=learning_rate)
    rng = np.random.default_rng(seed)
    completed, updates = (0, [])
    identity = encoder.signature()
    settings = dict(batch_size=batch_size, num_envs=num_envs, seed=seed, learning_rate=learning_rate,
                    reward_version=REWARD_VERSION, sampling_version=SAMPLING_VERSION)
    bind_identity(directory / 'identity.json', dict(encoder=identity, source=source_identity, settings=settings, policy=model.config.__dict__),
                  occupied=any((directory / n).exists() for n in ('pending.json', 'progress.json', 'current.json', 'result.json')))
    if has_resume:
        model, progress = restore_resume(directory, optimizer)
        if progress['encoder'] != identity:
            raise ValueError('RL resume encoder changed')
        if progress.get('source_identity') != source_identity:
            raise ValueError('RL resume source checkpoint changed')
        if progress.get('settings') != settings:
            raise ValueError('RL resume hyperparameters changed')
        completed, updates = (progress['episodes'], progress['updates'])
        rng.bit_generator.state = progress['rng']
    validate_rl_budget(episodes, batch_size, completed)
    model.enable_qat()
    model.train()
    def export():
        save_model(model, directory / 'model', packed=True, metadata=dict(rl_episodes=completed, rl_batches=len(updates), rl_updates=sum(u['update_applied'] for u in updates), qat_during_rl=True, reward_version=REWARD_VERSION, sampling_version=SAMPLING_VERSION))
        write_json(directory / 'result.json', dict(complete=completed == episodes, episodes=completed, updates=updates, actual_updates=sum(u['update_applied'] for u in updates), qat_during_rl=True, qat_mode='mxfp8', base_quantized=False, reward_version=REWARD_VERSION, sampling_version=SAMPLING_VERSION))

    if completed == episodes:
        export()
        return model
    envs = Environments(num_envs)
    try:
        while completed < episodes:
            size = min(batch_size, episodes - completed)
            if size < 2:
                raise ValueError('RL final batch must have at least two episodes')
            pending_path = directory / 'pending.json'
            pending = json.loads(pending_path.read_text()) if pending_path.exists() else None
            experience_path = directory / 'experience' / f'batch_{completed:06d}.npz'
            if pending and pending['start'] == completed:
                experience = load_experiences(experience_path)
                if pending['count'] != size or len(experience) != size or pending.get('sha256') != file_hash(experience_path):
                    raise ValueError({'pending_batch_changed': str(experience_path)})
                rng.bit_generator.state = pending['rng']
            else:
                experience = []
                for start in range(completed, completed + size, num_envs):
                    ordinals = list(range(start, min(start + num_envs, completed + size)))
                    wave = rollout_wave(envs, model, encoder, [seed + 2000000 + i for i in ordinals], ordinals, rng=rng)
                    experience.extend(wave)
                    for ordinal, row in zip(ordinals, wave, strict=True):
                        write_json(directory / 'episodes' / f'episode_{ordinal:06d}.json', dict(result=row['result'], trace=row['trace']))
                save_experiences(experience_path, experience)
                write_json(pending_path, dict(start=completed, count=size, rng=rng.bit_generator.state, sha256=file_hash(experience_path)))
            metrics = reinforce_update(model, optimizer, experience)
            if metrics['update_applied']:
                model._metadata = {**getattr(model, '_metadata', {}), 'fp8_training_completed': True}
            completed += size
            record = dict(update=len(updates) + 1, completed_episodes=completed, **metrics)
            updates.append(record)
            print(json.dumps(record), flush=True)
            save_resume(model, optimizer, directory, dict(episodes=completed, updates=updates, rng=rng.bit_generator.state, encoder=identity, source_identity=source_identity, settings=settings))
            export()
    finally:
        envs.close()
    return model

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('check')
    c.add_argument('--output', type=Path, default=ROOT / 'data/mac_migration/checks')
    c.add_argument('--simulation', action='store_true')
    c = commands.add_parser('import')
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c = commands.add_parser('collect')
    c.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    c.add_argument('--quota', type=int, default=300)
    c.add_argument('--seed', type=int, default=20261003)
    for command in ('cache', 'bc', 'rl', 'evaluate', 'benchmark'):
        c = commands.add_parser(command)
        c.add_argument('--model', type=Path, required=True)
        c.add_argument('--output', type=Path, required=True)
        if command in ('cache', 'benchmark'):
            c.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
        if command in ('bc',):
            c.add_argument('--cache', type=Path, required=True)
        if command in ('cache', 'rl', 'evaluate'):
            c.add_argument('--profile', type=Path)
        if command in ('cache', 'bc', 'rl'):
            c.add_argument('--batch-size', type=int, default=8 if command != 'bc' else 32)
        if command == 'bc':
            c.add_argument('--steps', type=int, default=12000)
            c.add_argument('--eval-every', type=int, default=250)
        if command in ('rl', 'evaluate'):
            c.add_argument('--episodes', type=int, default=128 if command == 'rl' else 50)
            c.add_argument('--num-envs', type=int, default=4)
        if command == 'benchmark':
            c.add_argument('--repeats', type=int, default=3)
    for command in ('run', 'launch'):
        c = commands.add_parser(command)
        c.add_argument('--config', type=Path, required=True)
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    for key in ('batch_size', 'steps', 'eval_every', 'episodes', 'num_envs', 'repeats'):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    mx.set_memory_limit(8 * 1024 ** 3)
    mx.set_cache_limit(1024 ** 3)
    if args.command in ('run', 'launch'):
        from .automation import launch, run
        return (launch if args.command == 'launch' else run)(args.config)
    with exclusive_lock(GPU_LOCK):
        if old_jobs_active():
            raise RuntimeError('Original Qwen task active; use the unattended run command to wait safely')
        return dispatch(args)


def dispatch(args):
    if args.command == 'check':
        from .checks import run_checks
        result = run_checks(args.output, ROOT, args.simulation)
        print(json.dumps(result, indent=2))
        return result
    if args.command == 'collect':
        if args.quota < 1 or args.seed < 0:
            raise ValueError('Invalid collection quota/seed')
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        import ur_pose_worker as worker
        args.dataset.mkdir(parents=True, exist_ok=True)
        output = args.dataset / 'collection'
        output.mkdir(exist_ok=True)
        return worker.collect(argparse.Namespace(run=args.dataset, output=output, seed=args.seed, quota=args.quota, width=320, height=180, min_free_gb=10.0, max_retries=20, smoke=False))
    if args.command == 'import':
        model = import_torch_checkpoint(args.checkpoint, args.output)
        return
    if args.command == 'rl':
        preflight_rl_budget(args.output, args.episodes, args.batch_size)
    model, _ = load_model(args.model, training=args.command in ('bc', 'rl'))
    if args.command == 'bc':
        train, signature = load_cache(args.cache, 'train', model.config)
        val, _ = load_cache(args.cache, 'val', model.config)
        return train_bc(model, train, val, args.output, steps=args.steps, batch_size=args.batch_size, eval_every=args.eval_every, cache_identity=signature, source_identity=model_identity(args.model))
    if args.command == 'benchmark':
        return benchmark(model, args.dataset, args.output, args.repeats)
    encoder = encoder_for(model, args.profile)
    try:
        if args.command == 'cache':
            return build_cache(args.dataset, args.output, encoder, model.config, args.batch_size)
        if args.command == 'evaluate':
            return evaluate(model, encoder, args.output, args.episodes, args.num_envs)
        if args.command == 'rl':
            return train_rl(model, encoder, args.output, args.episodes, args.batch_size, args.num_envs, source_identity=model_identity(args.model))
    finally:
        encoder.close()

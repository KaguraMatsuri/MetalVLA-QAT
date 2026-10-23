"""Audit regressions using temporary artifacts; never load a Qwen backbone."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from metalvla_qat.mac import automation, cli, data, train
from metalvla_qat.mac.model import MacPolicy, PolicyConfig, load_model, save_model
from metalvla_qat.mac.checks import assert_close
from metalvla_qat.mac.state import bind_identity, exclusive_lock, model_identity
import run_ur_qwen_mlx_rl as wrapper


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = PolicyConfig(feature_dim=32, horizon=2, execute_steps=2)
        self.model = MacPolicy(self.config)
        self.optimizer = optim.AdamW(learning_rate=1e-3)

    def test_torn_save_keeps_committed_generation(self):
        train.save_resume(self.model, self.optimizer, self.root, dict(step=0))
        initial = np.asarray(self.model.pose_projection.bias).copy()
        self.model.pose_projection.bias = self.model.pose_projection.bias + .5
        original = train.write_json
        for failed_name in ("progress.json", "manifest.json", "current.json"):
            def fail(path, value):
                if Path(path).name == failed_name:
                    raise OSError("injected")
                original(path, value)
            with self.subTest(failed_name=failed_name), patch.object(train, "write_json", side_effect=fail):
                with self.assertRaises(OSError):
                    train.save_resume(self.model, self.optimizer, self.root, dict(step=1))
            restored, progress = train.restore_resume(self.root, optim.AdamW(learning_rate=1e-3))
            self.assertEqual(progress["step"], 0)
            np.testing.assert_array_equal(restored.pose_projection.bias, initial)
        original_save = mx.save_safetensors
        def fail_optimizer(path, value):
            if Path(path).name == "optimizer.safetensors":
                raise OSError("injected")
            original_save(path, value)
        with patch.object(mx, "save_safetensors", side_effect=fail_optimizer):
            with self.assertRaises(OSError):
                train.save_resume(self.model, self.optimizer, self.root, dict(step=1))
        self.assertEqual(train.restore_resume(self.root, optim.AdamW(learning_rate=1e-3))[1]["step"], 0)
        train.save_resume(self.model, self.optimizer, self.root, dict(step=1))
        self.assertEqual(train.restore_resume(self.root, optim.AdamW(learning_rate=1e-3))[1]["step"], 1)
        self.assertEqual(len(list((self.root / "generations").iterdir())), 2)

    def test_corrupt_resume_rejected(self):
        train.save_resume(self.model, self.optimizer, self.root, dict(step=0))
        generation = train.resume_directory(self.root)
        data.write_json(generation / "progress.json", dict(step=999))
        with self.assertRaises(ValueError):
            train.restore_resume(self.root, optim.AdamW(learning_rate=1e-3))

    def test_legacy_resume_rejected(self):
        data.write_json(self.root / "progress.json", dict(step=0))
        with self.assertRaises(ValueError):
            train.resume_exists(self.root)

    def test_backbone_survives_generation_cleanup(self):
        backbone = self.root / "base.safetensors"
        mx.save_safetensors(str(backbone), dict(dummy=mx.ones((2,))))
        model = MacPolicy(PolicyConfig(feature_dim=32, horizon=2, execute_steps=2,
                                      backbone_sha256=data.file_hash(backbone)))
        model._backbone_file = backbone
        out = self.root / "resume"
        train.save_resume(model, self.optimizer, out, dict(step=0))
        model, _ = train.restore_resume(out, self.optimizer)
        for step in range(1, 5):
            train.save_resume(model, self.optimizer, out, dict(step=step))
        self.assertEqual(train.restore_resume(out, self.optimizer)[1]["step"], 4)
        self.assertEqual(len(list((out / "generations").iterdir())), 2)

    def test_identity_change_rejected(self):
        path = self.root / "identity.json"
        bind_identity(path, dict(steps=1))
        with self.assertRaises(ValueError):
            bind_identity(path, dict(steps=10000))
        with self.assertRaises(ValueError):
            bind_identity(self.root / "missing.json", {}, occupied=True)

    def test_process_lock_and_release(self):
        lock = self.root / "job.lock"
        code = "from metalvla_qat.mac.state import exclusive_lock; import sys\nwith exclusive_lock(sys.argv[1]): pass\n"
        with exclusive_lock(lock):
            result = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(lock.exists())

    def test_bc_lock_precedes_model_load(self):
        lock = self.root / "gpu.lock"
        with exclusive_lock(lock), patch.object(cli, "GPU_LOCK", lock), patch.object(cli, "load_model") as load:
            with self.assertRaises(RuntimeError):
                cli.main(["bc", "--model", "missing", "--cache", "missing", "--output", str(self.root)])
            load.assert_not_called()

    def test_parity_rejects_nonfinite_results(self):
        for a, b in (([np.nan], [0.]), ([np.nan], [np.nan]),
                     ([np.inf], [np.inf]), ([-np.inf], [-np.inf])):
            with self.subTest(a=a, b=b), np.errstate(invalid="ignore"):
                with self.assertRaises(AssertionError):
                    assert_close(a, b)
        self.assertEqual(assert_close([1., 2.], [1., 2.]), 0.)
        self.assertLessEqual(assert_close([1.], [1.+1e-6]), 1e-5)
        with self.assertRaises(AssertionError):
            assert_close([1.], [1.1])

    def test_rl_rejects_single_tail_before_model_load_or_rollout(self):
        encoder = SimpleNamespace(signature=lambda: {"fixture": True})
        for episodes, batch_size in ((1, 8), (9, 8), (129, 8), (3, 2), (8, 1)):
            with self.subTest(episodes=episodes, batch_size=batch_size):
                directory = self.root / f"invalid-{episodes}-{batch_size}"
                args = SimpleNamespace(command="rl", model="missing", output=directory,
                                       episodes=episodes, batch_size=batch_size)
                with patch.object(cli, "load_model") as load:
                    with self.assertRaises(ValueError):
                        cli.dispatch(args)
                    load.assert_not_called()
                with patch.object(cli, "Environments") as envs:
                    with self.assertRaises(ValueError):
                        cli.train_rl(self.model, encoder, directory, episodes=episodes, batch_size=batch_size)
                    envs.assert_not_called()
                self.assertFalse(directory.exists())

    def test_automation_rejects_invalid_rl_budget_before_binding_or_stages(self):
        out = self.root / "invalid-run"
        out.mkdir()
        source = self.root / "source.pt"
        source.write_bytes(b"fixture")
        dataset, _ = self.dataset()
        config = dict(rl_episodes=9, source_checkpoint=str(source), dataset=str(dataset))
        with patch.object(automation, "bind_identity") as bind, \
             patch.object(automation.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as child:
            with self.assertRaises(ValueError):
                automation._run(config, out)
            bind.assert_not_called()
            child.assert_not_called()

    def test_automation_preflight_does_not_import_mlx(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "from metalvla_qat.mac import automation; "
                "assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "src")], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rl_preflight_reads_only_verified_committed_progress(self):
        directory = self.root / "preflight"
        train.save_resume(self.model, self.optimizer, directory, dict(episodes=6))
        self.assertEqual(cli.preflight_rl_budget(directory, 9, 8), 6)
        train.save_resume(self.model, self.optimizer, directory, dict(episodes=8))
        args = SimpleNamespace(command="rl", model="missing", output=directory, episodes=9, batch_size=8)
        with patch.object(cli, "load_model") as load:
            with self.assertRaises(ValueError):
                cli.dispatch(args)
            load.assert_not_called()
        generation = train.resume_directory(directory)
        data.write_json(generation / "progress.json", dict(episodes=6))
        with self.assertRaises(ValueError):
            cli.preflight_rl_budget(directory, 9, 8)

    def test_rl_preflight_preserves_pending_batch_replay_and_integrity(self):
        encoder = SimpleNamespace(signature=lambda: {"fixture": True})
        experiences = [dict(result=dict(success=False), decisions=[], trace=[]) for _ in range(2)]
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt):
                directory = self.root / f"pending-{corrupt}"
                with patch.object(cli, "Environments"), patch.object(cli, "rollout_wave", return_value=experiences), \
                     patch.object(cli, "reinforce_update", side_effect=OSError("injected")):
                    with self.assertRaises(OSError):
                        cli.train_rl(self.model, encoder, directory, episodes=2, batch_size=2)
                pending_path = directory / "pending.json"
                pending = json.loads(pending_path.read_text())
                if corrupt:
                    pending["count"] = 3
                    data.write_json(pending_path, pending)
                with patch.object(cli, "Environments"), patch.object(cli, "rollout_wave") as rollout, \
                     patch.object(cli, "reinforce_update", return_value=dict(update_applied=False)) as update:
                    if corrupt:
                        with self.assertRaises(ValueError):
                            cli.train_rl(self.model, encoder, directory, episodes=2, batch_size=2)
                        update.assert_not_called()
                    else:
                        cli.train_rl(self.model, encoder, directory, episodes=2, batch_size=2)
                        update.assert_called_once()
                    rollout.assert_not_called()

    def test_rl_tail_validation_uses_committed_remaining_progress(self):
        encoder = SimpleNamespace(signature=lambda: {"fixture": True})
        for completed, target, expected in ((6, 9, 3), (8, 9, None), (8, 8, 0), (0, 10, 10)):
            with self.subTest(completed=completed, target=target):
                directory = self.root / f"remaining-{completed}-{target}"
                seen = []

                def wave(envs, model, encoder, seeds, ordinals, **kwargs):
                    seen.extend(ordinals)
                    return [dict(result=dict(success=False), decisions=[], trace=[]) for _ in ordinals]

                with patch.object(cli, "Environments"), patch.object(cli, "rollout_wave", side_effect=wave), \
                     patch.object(cli, "reinforce_update", return_value=dict(update_applied=False)):
                    if completed:
                        cli.train_rl(self.model, encoder, directory, episodes=completed, batch_size=8)
                    seen.clear()
                    if expected is None:
                        pointer = (directory / "current.json").read_bytes()
                        with self.assertRaises(ValueError):
                            cli.train_rl(self.model, encoder, directory, episodes=target, batch_size=8)
                        self.assertEqual((directory / "current.json").read_bytes(), pointer)
                    else:
                        cli.train_rl(self.model, encoder, directory, episodes=target, batch_size=8)
                    self.assertEqual(len(seen), expected or 0)

    def test_visible_rollout_stops_at_first_success(self):
        import torch
        import ur_pose_worker as worker

        class Task:
            def reset(self, *args):
                self.steps = 0
                self.success = False
                self.trace = []

            def observe(self):
                return dict(images=np.zeros((3, 4, 4, 3), np.uint8),
                            state=np.zeros(8, np.float32), previous=np.zeros(8, np.float32))

            def execute(self, *args):
                self.steps += 1
                self.success = self.steps == self.success_step
                self.trace.append(dict(tcp=[0., 0., 0.], cube=[1., 0., 0.]))

            def score(self, *args):
                return dict(success=self.success, score=float(self.success), bilateral_grasp=False,
                            lifted=False, placement_error_m=1.)

        class Policy:
            config = SimpleNamespace(horizon=3, execute_steps=3, image_size=64)
            vocabulary = {"fixture": 1}

            def parameters(self):
                yield torch.zeros(1)

            def empty_history(self, *args):
                return None, None

            def __call__(self, *args):
                return torch.zeros((1, 3, 6)), None, None, None

            def execution_prefix(self, actions):
                return actions

            def decode(self, actions):
                return torch.zeros((1, 3, 3)), torch.zeros((1, 3)), torch.zeros((1, 3), dtype=torch.bool)

        for success_step, expected_steps in ((2, 2), (0, 3)):
            task = Task()
            task.success_step = success_step
            args = SimpleNamespace(max_decisions=1, grasp_retries=0, precision_reward=False)
            with self.subTest(success_step=success_step), patch.object(worker, "budget"):
                result, _ = worker.rollout(args, task, Policy(), 1, 0, 0)
                self.assertEqual(task.steps, expected_steps)
                self.assertEqual(result["success"], bool(success_step))

    def dataset(self):
        root = self.root / "dataset"
        (root / "raw").mkdir(parents=True)
        rows = []
        for i, split in enumerate(("train", "val")):
            path = root / f"raw/{i}.npz"
            data.write_npz(path, dict(desired=np.zeros((3, 5), np.float32), state=np.zeros((3, 8), np.float32),
                                     previous=np.zeros((3, 8), np.float32), images=np.zeros((3, 3, 4, 4, 3), np.uint8)))
            rows.append(dict(file=f"raw/{i}.npz", sha256=data.file_hash(path), schema="ur5e_pick_place_pose_v1",
                             success=True, split=split, region=i, ordinal=i))
        (root / "raw/manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        return root, rows

    def test_partial_cache_recovery_and_integrity(self):
        root, rows = self.dataset()
        encoder = SimpleNamespace(signature=lambda: {"test": True}, hidden_size=32,
                                  prepare=lambda raw, instruction: raw,
                                  encode_prepared=lambda raw: np.zeros((len(raw), 32), np.float32))
        out = self.root / "cache"
        original = data.write_npz
        calls = []
        def interrupted(path, value):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("injected")
            original(path, value)
        with patch.object(data, "write_npz", side_effect=interrupted):
            with self.assertRaises(OSError):
                data.build_cache(root, out, encoder, self.config)
        self.assertTrue((out / "signature.json").exists())
        self.assertFalse((out / "complete.json").exists())
        first_hash = data.file_hash(data.episode_path(out, rows[0]))
        data.build_cache(root, out, encoder, self.config)
        self.assertEqual(first_hash, data.file_hash(data.episode_path(out, rows[0])))
        self.assertEqual(len(data.load_cache(out, "val", self.config)[0]), 1)
        data.episode_path(out, rows[0]).write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            data.load_cache(out, "train", self.config)

    def test_evaluation_seed_and_sequence(self):
        encoder = SimpleNamespace(signature=lambda: {"test": True})
        envs = SimpleNamespace(close=lambda: None)
        def wave(envs, model, encoder, seeds, ordinals):
            return [dict(result=dict(seed=s, ordinal=i, region=i % 9, success=False, score=0, grasp_retries=0), trace=[])
                    for s, i in zip(seeds, ordinals)]
        with patch.object(cli, "Environments", return_value=envs), patch.object(cli, "rollout_wave", side_effect=wave) as roll:
            cli.evaluate(self.model, encoder, self.root, episodes=2, seed=100)
            with self.assertRaises(ValueError):
                cli.evaluate(self.model, encoder, self.root, episodes=2, seed=900)
            self.assertEqual(roll.call_count, 1)
            cli.evaluate(self.model, encoder, self.root, episodes=3, seed=100)
            self.assertEqual(roll.call_count, 2)
            cli.evaluate(self.model, encoder, self.root, episodes=3, seed=100)
            self.assertEqual(roll.call_count, 2)

    def test_automation_path_cache_and_config_identity(self):
        dataset, _ = self.dataset()
        source = self.root / "source.pt"
        source.write_bytes(b"source")
        out = self.root / "run"
        config = dict(output=str(out), source_checkpoint=str(source), dataset=str(dataset), bc_steps=1, rl_episodes=8)
        config_path = self.root / "config.json"
        data.write_json(config_path, config)
        called = []
        def child(arguments, **kwargs):
            self.assertTrue(all(isinstance(v, str) for v in arguments))
            stage = arguments[2]
            called.append(stage)
            target = Path(arguments[arguments.index("--output")+1])
            if stage == "import":
                self.model._metadata = dict(source_sha256=data.file_hash(source))
                save_model(self.model, target)
            elif stage == "cache":
                data.write_json(target / "signature.json", {})
                data.write_json(target / "complete.json", {})
            elif stage == "bc":
                save_model(self.model, target / "best_master")
                save_model(self.model, target / "model", packed=True)
                data.write_json(target / "result.json", dict(complete=True))
            elif stage == "rl":
                save_model(self.model, target / "model", packed=True)
                data.write_json(target / "result.json", dict(complete=True))
            elif stage == "benchmark":
                data.write_json(target / "result.json", dict(latency_target_passed=False, throughput_target_passed=False))
            elif stage == "evaluate":
                data.write_json(target / "result.json", dict(complete=True, successes=0, results=[dict(placement_error_m=.1)]*50))
            return SimpleNamespace(returncode=0)
        with patch.object(automation.subprocess, "run", side_effect=child):
            automation.run(config_path)
            called.clear()
            automation.run(config_path)
            self.assertEqual(called, ["check", "cache"])
            data.write_json(config_path, dict(config, bc_steps=10000))
            called.clear()
            with self.assertRaises(ValueError):
                automation.run(config_path)
            with self.assertRaises(ValueError):
                automation.launch(config_path)
            self.assertEqual(called, [])

    def test_rl_final_export_after_interruption(self):
        encoder = SimpleNamespace(signature=lambda: {"test": True})
        out = self.root / "rl"
        envs = SimpleNamespace(close=lambda: None)
        experiences = [dict(result=dict(success=False), decisions=[], trace=[]) for _ in range(2)]
        with patch.object(cli, "Environments", return_value=envs), patch.object(cli, "rollout_wave", return_value=experiences), \
             patch.object(cli, "reinforce_update", return_value=dict(update_applied=True)), \
             patch.object(cli, "save_model", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                cli.train_rl(self.model, encoder, out, episodes=2, batch_size=2)
        self.assertTrue((out / "current.json").exists())
        with patch.object(cli, "Environments") as environments, patch.object(cli, "reinforce_update") as update:
            cli.train_rl(self.model, encoder, out, episodes=2, batch_size=2)
            environments.assert_not_called()
            update.assert_not_called()
        metadata = json.loads((out / "model/config.json").read_text())["metadata"]
        self.assertTrue(metadata["fp8_training_completed"])
        self.assertEqual(metadata["rl_updates"], 1)

    def test_bc_resume_matches_uninterrupted_updates(self):
        rng = np.random.default_rng(12)
        episode = dict(features=rng.normal(size=(3, 32)).astype(np.float32),
                       state=rng.normal(size=(3, 8)).astype(np.float32), previous=np.zeros((3, 8), np.float32),
                       target=rng.normal(size=(3, 2, 6)).astype(np.float32), valid=np.ones((3, 2), bool))
        source = self.root / "source"
        save_model(self.model, source)
        full, resumed = self.root / "full", self.root / "resumed"
        for directory, steps in ((full, 4), (resumed, 2), (resumed, 4)):
            model, _ = load_model(source, training=True)
            train.train_bc(model, [episode], [episode], directory, steps=steps, eval_every=1)
        a, pa = train.restore_resume(full, optim.AdamW(learning_rate=1e-4))
        b, pb = train.restore_resume(resumed, optim.AdamW(learning_rate=1e-4))
        from mlx.utils import tree_flatten
        for (_, x), (_, y) in zip(tree_flatten(a.parameters()), tree_flatten(b.parameters()), strict=True):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(pa, pb)
        self.assertTrue(a._metadata["fp8_training_completed"])

    def test_wrapper_evaluation_failure_resume_without_retraining(self):
        source = self.root / "source"
        save_model(self.model, source)
        out = self.root / "run"
        stages = []
        failed = False
        def child(arguments, **kwargs):
            nonlocal failed
            if "rl" in arguments:
                stages.append("rl")
                target = Path(arguments[arguments.index("--output")+1])
                save_model(self.model, target / "model", packed=True)
                data.write_json(target / "result.json", dict(complete=True, episodes=8, actual_updates=1))
            else:
                stages.append("evaluate")
                target = Path(arguments[arguments.index("--output-dir")+1])
                target.mkdir(exist_ok=True)
                if not failed:
                    failed = True
                    return SimpleNamespace(returncode=1)
                self.assertIn("--resume", arguments)
                data.write_json(target / "result.json", dict(complete=True, episodes=10, successes=0))
            return SimpleNamespace(returncode=0)
        with patch.object(wrapper.subprocess, "run", side_effect=child):
            with self.assertRaises(subprocess.CalledProcessError):
                wrapper.run(source, out, episodes=8)
            wrapper.run(source, out, episodes=8, resume=True)
        self.assertEqual(stages, ["rl", "evaluate", "evaluate"])

    def test_visual_evaluator_resumes_completed_episodes(self):
        import quick_eval_unfrozen as visual
        source = self.root / "source"
        save_model(self.model, source)
        args = SimpleNamespace(model=source, checkpoint=None, output_dir=self.root / "eval",
                               resume=False, seed=100, device="cpu", close_after=0)
        task = SimpleNamespace(model=None, data=None, execute=lambda *a: None, trace=[], close=lambda: None)
        viewer = MagicMock()
        viewer.__enter__.return_value.is_running.return_value = False
        calls = []
        def rollout(args, task, model, seed, region, ordinal):
            calls.append(ordinal)
            if calls == [0, 1, 2]:
                raise OSError("injected")
            return dict(success=False, score=0, grasp_retries=0, failure_type="test"), None
        with patch.object(visual, "URTask", return_value=task), patch.object(visual, "VisiblePolicy"), \
             patch.object(visual.mujoco.viewer, "launch_passive", return_value=viewer), \
             patch.object(visual.worker, "rollout", side_effect=rollout):
            with self.assertRaises(OSError):
                visual.evaluate(args)
            args.resume = True
            visual.evaluate(args)
            visual.evaluate(args)
            self.assertEqual(calls, [0, 1, 2, *range(2, 10)])
            args.seed = 900
            with self.assertRaises(ValueError):
                visual.evaluate(args)


if __name__ == "__main__":
    unittest.main()

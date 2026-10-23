"""Reward and action-distribution regressions without loading Qwen."""

import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from metalvla_qat.score import REWARD_VERSION, apply_reward, shaped_reward
from metalvla_qat.mac.distribution import SAMPLING_VERSION, gripper_logits
from metalvla_qat.mac.runtime import sample_actions
from metalvla_qat.mac.train import action_log_probability, reinforce_update
from metalvla_qat.mac import cli, data
from metalvla_qat.mac.model import MacPolicy, PolicyConfig
from ur_pose_worker import precision_score
from ur_pick_place import URTask


class RewardTests(unittest.TestCase):
    def test_failed_rewards_remain_distinct_below_zero(self):
        a = apply_reward(dict(success=False, decisions=3, grasp_retries=2))
        b = apply_reward(dict(success=False, decisions=3, grasp_retries=2, grasp_alignment_best_m=.1))
        self.assertEqual(a["score"], b["score"])
        self.assertLess(a["training_reward"], b["training_reward"])
        self.assertLess(b["training_reward"], 0)
        self.assertEqual(a["training_reward"], -.9)

    def test_historical_contact_alias(self):
        old = apply_reward(dict(success=False, bilateral_grasp=True))
        current = apply_reward(dict(success=False, bilateral_grasp=True, ever_grasped=True))
        self.assertEqual(old["training_reward"], current["training_reward"])
        self.assertEqual(old["reward_components"], current["reward_components"])
        self.assertAlmostEqual(current["training_reward"], .65 + .8)

    def test_longer_success_path_never_scores_higher(self):
        row = dict(success=True, ideal_path_m=.5, placement_error_m=.01,
                   grasp_center_error_m=.01, elapsed_sim_s=5, decisions=1)
        values = [apply_reward(dict(row, tcp_path_m=p))["training_reward"] for p in (.25, .5, 1., 2.)]
        self.assertEqual(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertGreater(values[2], values[3])
        with self.assertRaises(KeyError):
            shaped_reward(dict(success=True))

    def test_ur_result_provides_reward_fields(self):
        task = SimpleNamespace(start=np.array([0., 0., 0.]), goal=np.array([.3, 0., 0.]),
            path=1., elapsed=5., grasp_center_error=None, ever_grasped=True, lifted=False,
            success=False, stable=0., cube_position=lambda: np.zeros(3), tcp_position=lambda: np.ones(3))
        result = URTask.score(task, 3)
        self.assertTrue(result["ever_grasped"])
        self.assertTrue(result["bilateral_grasp"])
        self.assertAlmostEqual(result["ideal_path_m"], .7)

    def test_visible_and_batch_use_same_scoring(self):
        row = dict(success=False, score=.2, bilateral_grasp=True, ever_grasped=True,
                   lifted=False, grasp_center_error_m=.02)
        task = SimpleNamespace(score=lambda decisions: dict(row, decisions=decisions))
        result = precision_score(task, 3, 2, .04)
        expected = apply_reward(dict(row, decisions=3, grasp_retries=2, grasp_alignment_best_m=.04))
        for field in ("score", "training_reward", "reward_version", "reward_components"):
            self.assertEqual(result[field], expected[field])

    def test_sampling_frequency_matches_log_probability(self):
        reference = np.zeros((100000, 6), np.float32)
        reference[:, 4], reference[:, 5] = 1., .8
        _, latent, closed = sample_actions(reference, np.random.default_rng(21))
        probability = 1 / (1 + np.exp(-gripper_logits(np.array(.8), .5, np)))
        self.assertLess(abs(closed.mean() - probability), .003)
        scales = np.array([.10, .10, .06, .08]) * .5
        expected = np.sum(-.5*(latent/scales)**2-np.log(scales)-.5*math.log(2*math.pi), axis=-1)
        expected += np.where(closed > 0, np.log(probability), np.log1p(-probability))
        actual = np.asarray(action_log_probability(mx.array(reference), mx.array(latent), mx.array(closed)))
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_confident_gripper_holds_full_chunk(self):
        reference = np.zeros((1000, 130, 6), np.float32)
        reference[..., 4] = 1.
        for direction in (-1., 1.):
            reference[..., 5] = direction
            _, _, closed = sample_actions(reference, np.random.default_rng(33))
            self.assertGreater(np.mean(np.all(closed == (direction > 0), axis=1)), .999)

    def test_logits_and_gradients_finite_at_saturation(self):
        actions = mx.array([-1., -.999, -.5, 0., .5, .999, 1.])
        logits = gripper_logits(actions, .5, mx)
        grads = mx.grad(lambda x: mx.sum(gripper_logits(x, .5, mx)))(actions)
        self.assertTrue(np.isfinite(logits).all())
        self.assertTrue(np.isfinite(grads).all())
        for scale in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                gripper_logits(actions, scale, mx)

    def test_legacy_experiences_rejected_and_training_reward_used(self):
        model = MacPolicy(PolicyConfig(feature_dim=32, horizon=2, execute_steps=2))
        optimizer = optim.AdamW(learning_rate=1e-4)
        with self.assertRaises(ValueError):
            reinforce_update(model, optimizer, [dict(result=dict(score=0), decisions=[])]*2)
        experiences = [dict(result=dict(score=s, training_reward=-.2, reward_version=REWARD_VERSION,
                         sampling_version=SAMPLING_VERSION), decisions=[]) for s in (0, 10)]
        result = reinforce_update(model, optimizer, experiences)
        self.assertFalse(result["update_applied"])
        self.assertAlmostEqual(result["mean_reward"], -.2)

    def test_old_run_identity_rejected_before_rollout(self):
        model = MacPolicy(PolicyConfig(feature_dim=32, horizon=2, execute_steps=2))
        encoder = SimpleNamespace(signature=lambda: {"test": True})
        with tempfile.TemporaryDirectory() as directory:
            data.write_json(Path(directory)/"identity.json", dict(encoder=encoder.signature(), source=None,
                settings=dict(batch_size=8, num_envs=4, seed=20261008, learning_rate=1e-6), policy=model.config.__dict__))
            with patch.object(cli, "Environments") as environments:
                with self.assertRaises(ValueError):
                    cli.train_rl(model, encoder, directory)
                environments.assert_not_called()


if __name__ == "__main__":
    unittest.main()

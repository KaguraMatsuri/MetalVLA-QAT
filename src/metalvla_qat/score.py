"""Shared reward: physical precision and efficiency, explicit retry accounting."""

import math
import numpy as np

REWARD_VERSION = 2


def shaped_reward(result, penalize_replanning=True):
    def quality(value, scale):
        return 0. if value is None or not math.isfinite(float(value)) else math.exp(-float(value)/scale)

    alignment = quality(result.get("grasp_alignment_best_m"), .025)
    center = quality(result.get("grasp_center_error_m"), .015)
    place = quality(result.get("placement_error_m"), .020)
    decisions = max(1, int(result.get("decisions", 1)))
    if result.get("success"):
        ideal = float(result["ideal_path_m"])
        path = max(float(result.get("tcp_path_m", 0.)), .001)
        if not math.isfinite(ideal) or ideal <= 0:
            raise ValueError({"ideal_path_m": ideal})
        efficiency = min(1., ideal/max(ideal, path))
        elapsed = max(float(result.get("elapsed_sim_s", 5.)), 5.)
        raw = 6.5+1.25*center+1.5*place+.3*efficiency+.2*min(1., 5./elapsed)
        maximum = 10.
    else:
        raw = (.9*alignment+.65*float(result.get("bilateral_grasp", False))
               +.8*float(result.get("ever_grasped", result.get("bilateral_grasp", False)))
               +1.5*float(result.get("lifted", False))+.5*center)
        if result.get("lifted"):
            raw += 1.4*place
        maximum = 5.5
    charged = decisions-1 if penalize_replanning else int(result.get("grasp_retries", 0))
    penalty = .45*charged
    return float(np.clip(raw-penalty, 0., maximum)), dict(
        reward_version=REWARD_VERSION, training_reward=float(raw-penalty),
        alignment_quality=alignment, grasp_center_quality=center, placement_quality=place,
        policy_decisions=decisions, reinference_count=decisions-1, reinference_penalty=penalty,
        charged_retries=charged, normal_replanning_penalized=penalize_replanning)


def apply_reward(result):
    score, components = shaped_reward(result, penalize_replanning=False)
    result.update(score=score, training_reward=components["training_reward"],
                  reward_version=REWARD_VERSION, reward_components=components)
    return result

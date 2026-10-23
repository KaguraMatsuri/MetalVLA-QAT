"""Floating Qwen decoder-MLP fine-tuning for export into the MLX task policy."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .qwen_vla import BASE_MODEL, BASE_REVISION, QwenVLA

class QwenPoseVLA(nn.Module):
    """Qwen3.5-0.8B and the H130 fine-tuning supervision head."""
    policy_kind = 'qwen_ur_pose_v1_unfrozen'

    def __init__(self, vocabulary, horizon=130, execute_steps=130, image_size=192, history_length=0, workspace_low=(-0.39, -0.1, 0.732), workspace_high=(0.39, 0.42, 1.12), base_model=BASE_MODEL, base_revision=BASE_REVISION, backbone_dtype="float32"):
        super().__init__()
        if horizon < 1 or not 1 <= execute_steps <= horizon:
            raise ValueError('Invalid Qwen action horizon or execution prefix')
        if image_size != 192:
            raise ValueError('The pinned Qwen processor is configured for 192px images')
        if history_length != 0:
            raise ValueError(history_length)
        self.vocabulary = dict(vocabulary)
        self.encoder = QwenVLA(self.vocabulary, base_model=base_model, base_revision=base_revision, image_size=image_size, backbone_dtype=backbone_dtype)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        hidden = self.encoder.backbone.config.text_config.hidden_size
        self.feature_norm = nn.LayerNorm(hidden)
        self.state_encoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(hidden + 64, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, horizon * 6))
        self.position_origin = nn.Parameter(torch.tensor([(lo + hi) / 2 for lo, hi in zip(workspace_low, workspace_high)]), requires_grad=False)
        self.position_scale = nn.Parameter(torch.tensor([(hi - lo) / 2 for lo, hi in zip(workspace_low, workspace_high)]), requires_grad=False)
        self.config = SimpleNamespace(horizon=horizon, execute_steps=execute_steps, image_size=image_size, cameras=3, memory_steps=0, memory_width=hidden + 16, history_length=history_length, replan_after_chunk=bool(history_length), period_s=0.1, workspace_low=tuple(workspace_low), workspace_high=tuple(workspace_high))
        self.model_config = dict(vocabulary=self.vocabulary, horizon=horizon, execute_steps=execute_steps, image_size=image_size, history_length=history_length, workspace_low=tuple(workspace_low), workspace_high=tuple(workspace_high), base_model=base_model, base_revision=base_revision, backbone_dtype=backbone_dtype)
        self.qat_prepared = False

    def train(self, mode=True):
        super().train(mode)
        self.encoder.backbone.eval()
        return self

    def empty_history(self, batch, device=None):
        device = device or self.position_origin.device
        return (torch.zeros(batch, self.config.memory_steps, self.config.memory_width, device=device), torch.zeros(batch, self.config.memory_steps, dtype=torch.bool if self.config.history_length else torch.float32, device=device))

    def encode_instruction(self, tokens):
        return tokens

    def features(self, images, tokens):
        return self.encoder.extract_features(images, tokens)

    def actions_from_features(self, features, state, previous, history=None, valid=None):
        if state.ndim != 2 or state.shape[-1] != 8 or previous.shape != state.shape:
            raise ValueError('Expected aligned 8-D robot state and previous control')
        robot = torch.cat((state.float(), previous.float()), -1)
        state_input = F.pad(robot, (0, 16))
        encoded = self.state_encoder(state_input)
        fused = torch.cat((self.feature_norm(features.float()), encoded), -1)
        raw = self.action_head(fused).reshape(-1, self.config.horizon, 6)
        return torch.cat((raw[..., :3].tanh(), raw[..., 3:5], raw[..., 5:6].tanh()), -1)

    def forward(self, images, tokens, state, previous, history=None, valid=None):
        features = self.features(images, tokens)
        if history is None or valid is None:
            history, valid = self.empty_history(len(state), state.device)
        actions = self.actions_from_features(features, state, previous, history, valid)
        return (actions, state.new_zeros((len(state), 3)), history, valid)

    def execution_prefix(self, actions):
        if actions.shape[-2:] != (self.config.horizon, 6):
            raise ValueError('Unexpected Qwen pose action chunk')
        return actions[..., :self.config.execute_steps, :]

    def decode(self, actions):
        if actions.ndim < 2 or actions.shape[-1] != 6 or (not torch.isfinite(actions).all()):
            raise ValueError('Expected finite Qwen pose actions [...,H,6]')
        if (actions[..., :3].abs() > 1.00001).any():
            raise ValueError('Qwen predicted a position outside the normalized workspace')
        yaw_vector = actions[..., 3:5]
        if (torch.linalg.vector_norm(yaw_vector, dim=-1) < 1e-06).any():
            raise ValueError('Undefined yaw: sin/cos vector is zero')
        xyz = actions[..., :3] * self.position_scale + self.position_origin
        yaw = torch.atan2(yaw_vector[..., 0], yaw_vector[..., 1])
        closed = actions[..., 5] >= 0
        return (xyz, yaw, closed)

    def targets(self, desired):
        """World XYZ/yaw/binary-gripper labels -> normalized six-value actions."""
        xyz = (desired[..., :3] - self.position_origin) / self.position_scale
        yaw = desired[..., 3]
        gripper = desired[..., 4] * 2 - 1
        return torch.cat((xyz, torch.sin(yaw).unsqueeze(-1), torch.cos(yaw).unsqueeze(-1), gripper.unsqueeze(-1)), -1)

def pose_loss(prediction, target):
    """Smooth pose regression plus binary-gripper regression over a full chunk."""
    xyz = F.smooth_l1_loss(prediction[..., :3].contiguous(), target[..., :3].contiguous())
    yaw = F.smooth_l1_loss(prediction[..., 3:5].contiguous(), target[..., 3:5].contiguous())
    gripper = F.smooth_l1_loss(prediction[..., 5].contiguous(), target[..., 5].contiguous())
    return xyz + yaw + gripper

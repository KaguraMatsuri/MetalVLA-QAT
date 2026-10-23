"""Pure MLX pose policy with official MXFP8 training and packed snapshots."""
from dataclasses import asdict, dataclass, replace
import json
import hashlib
import os
import shutil
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .data import file_hash
LINEARS = ('state_projection', 'fusion_0', 'fusion_1', 'pose_projection')

@dataclass(frozen=True)
class PolicyConfig:
    feature_dim: int = 1024
    base_model: str = 'Qwen/Qwen3.5-0.8B'
    base_revision: str = '2fc06364715b967f1860aea9cf38778875588b17'
    horizon: int = 130
    execute_steps: int = 130
    history_length: int = 0
    image_size: int = 192
    workspace_low: tuple = (-0.39, -0.1, 0.732)
    workspace_high: tuple = (0.39, 0.42, 1.12)
    period_s: float = 0.1
    backbone_sha256: str | None = None

    def __post_init__(self):
        if self.feature_dim < 1 or self.feature_dim % 32:
            raise ValueError('feature_dim must be positive and divisible by 32')
        if self.history_length != 0 or not 1 <= self.execute_steps <= self.horizon:
            raise ValueError('Invalid history or execution prefix')
        if len(self.workspace_low) != 3 or any((a >= b for a, b in zip(self.workspace_low, self.workspace_high))):
            raise ValueError('Invalid workspace')

class BiasedLinear(nn.Module):
    """Official QQLinear plus a floating bias; no custom quantization kernel."""

    def __init__(self, inputs, outputs, fp8=False):
        super().__init__()
        self._outputs = outputs
        padded_outputs = (outputs + 31) // 32 * 32
        self.core = nn.QQLinear(inputs, padded_outputs, mode='mxfp8', group_size=32, bits=8) if fp8 else nn.Linear(inputs, outputs, bias=False)
        if fp8 and padded_outputs != outputs:
            self.core.weight = mx.concatenate((self.core.weight[:outputs], mx.zeros((padded_outputs - outputs, inputs))), axis=0)
        self.bias = mx.zeros((outputs,))

    def __call__(self, x):
        if isinstance(self.core, nn.QQLinear):
            shape = x.shape
            values = x.reshape(-1, shape[-1])
            rows = values.shape[0]
            if self.core.training and rows % 32:
                values = mx.pad(values, ((0, 32 - rows % 32), (0, 0)))
            result = self.core(values)[:rows, :self._outputs]
            return result.reshape(*shape[:-1], self._outputs) + self.bias
        return self.core(x) + self.bias

class MacPolicy(nn.Module):

    def __init__(self, config=PolicyConfig(), fp8=False):
        super().__init__()
        self._config = config
        self._fp8 = fp8
        h = config.feature_dim
        self.feature_norm = nn.LayerNorm(h, eps=1e-05)
        self.state_projection = BiasedLinear(32, 64, fp8)
        self.fusion_0 = BiasedLinear(h + 64, 512, fp8)
        self.fusion_1 = BiasedLinear(512, 128, fp8)
        self.pose_projection = BiasedLinear(128, config.horizon * 6, fp8)
        lo, hi = (mx.array(config.workspace_low), mx.array(config.workspace_high))
        self._origin = (lo + hi) / 2
        self._scale = (hi - lo) / 2

    @property
    def config(self):
        return self._config

    @property
    def fp8(self):
        return self._fp8

    @property
    def linear_names(self):
        return LINEARS

    def empty_history(self, batch):
        width = self.config.feature_dim + 16
        return (mx.zeros((batch, self.config.history_length, width)), mx.zeros((batch, self.config.history_length), dtype=mx.bool_))

    def actions(self, features, state, previous, history=None, valid=None):
        batch = features.shape[0]
        if features.shape != (batch, self.config.feature_dim) or state.shape != (batch, 8) or previous.shape != state.shape:
            raise ValueError('Feature/state contract mismatch')
        robot = mx.concatenate((state, previous), axis=-1)
        state_input = mx.pad(robot, ((0, 0), (0, 16)))
        encoded = nn.relu(self.state_projection(state_input))
        fused = mx.concatenate((self.feature_norm(features), encoded), axis=-1)
        fused = nn.relu(self.fusion_0(fused))
        fused = nn.relu(self.fusion_1(fused))
        raw = self.pose_projection(fused).reshape(batch, self.config.horizon, 6)
        return mx.concatenate((mx.tanh(raw[..., :3]), raw[..., 3:5], mx.tanh(raw[..., 5:6])), axis=-1)

    def __call__(self, features, state, previous, history=None, valid=None):
        if history is None:
            history, valid = self.empty_history(len(features))
        actions = self.actions(features, state, previous, history, valid)
        return (actions, history, valid)

    def decode(self, actions):
        import numpy as np
        a = np.asarray(actions)
        if a.shape[-1] != 6 or not np.isfinite(a).all() or np.abs(a[..., :3]).max() > 1.00001:
            raise ValueError('Invalid normalized pose chunk')
        if np.any(np.linalg.norm(a[..., 3:5], axis=-1) < 1e-06):
            raise ValueError('Undefined yaw')
        return (a[..., :3] * np.asarray(self._scale) + np.asarray(self._origin), np.arctan2(a[..., 3], a[..., 4]), a[..., 5] >= 0)

    def enable_qat(self):
        if self.fp8:
            return
        for name in self.linear_names:
            layer = getattr(self, name)
            old = layer.core
            padded_outputs = (layer._outputs + 31) // 32 * 32
            aligned = nn.Linear(old.weight.shape[1], padded_outputs, bias=False)
            aligned.weight = mx.pad(old.weight, ((0, padded_outputs - layer._outputs), (0, 0)))
            layer.core = nn.QQLinear.from_linear(aligned, mode='mxfp8', group_size=32, bits=8)
        self._fp8 = True
        self.train()

    def snapshot(self):
        """Never eval() a training master: QQLinear eval quantizes in-place."""
        mx.eval(self.parameters())
        copy = MacPolicy(self.config, self.fp8)
        copy._backbone_file = getattr(self, '_backbone_file', None)
        copy._metadata = getattr(self, '_metadata', {})
        already_packed = self.fp8 and all((getattr(getattr(self, name).core, '_quantized', False) for name in self.linear_names))
        if already_packed:
            copy.eval()
        copy.load_weights([(name, mx.array(value)) for name, value in tree_flatten(self.parameters())], strict=True)
        copy.eval()
        mx.eval(copy.parameters())
        return copy

def save_model(model, directory, packed=False, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if model.config.backbone_sha256:
        source = Path(model._backbone_file)
        destination = directory / 'qwen_mlp.safetensors'
        if source.resolve() != destination.resolve():
            temporary = directory / 'qwen_mlp.tmp.safetensors'
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        if file_hash(destination) != model.config.backbone_sha256:
            raise ValueError({'backbone_sha256': model.config.backbone_sha256})
    to_save = model.snapshot() if packed else model
    if not packed and any((getattr(getattr(model, name).core, '_quantized', False) for name in model.linear_names)):
        raise ValueError('Training master must not contain packed weights')
    mx.eval(to_save.parameters())
    temporary = directory / 'adapter.tmp.safetensors'
    mx.save_safetensors(str(temporary), dict(tree_flatten(to_save.parameters())))
    os.replace(temporary, directory / 'adapter.safetensors')
    payload = dict(schema='metalvla_mac_v1', config=asdict(model.config), fp8=model.fp8, packed=packed, metadata={**getattr(model, '_metadata', {}), **(metadata or {})})
    temporary = directory / 'config.tmp.json'
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, directory / 'config.json')
    quant = dict(mode='mxfp8' if model.fp8 else 'float32', bits=8 if model.fp8 else 32, group_size=32, layers=list(model.linear_names), base_quantized=False, training_row_alignment=32, physical_output_alignment=32, floating_parts=['Qwen', 'bias', 'normalization'])
    (directory / 'quantization.json').write_text(json.dumps(quant, indent=2) + '\n')

def load_model(directory, training=False):
    directory = Path(directory)
    payload = json.loads((directory / 'config.json').read_text())
    if payload.get('schema') != 'metalvla_mac_v1':
        raise ValueError('Unknown MLX checkpoint schema')
    model = MacPolicy(PolicyConfig(**payload['config']), fp8=payload['fp8'])
    model._metadata = payload['metadata']
    if model.config.backbone_sha256:
        model._backbone_file = directory / 'qwen_mlp.safetensors'
        if file_hash(model._backbone_file) != model.config.backbone_sha256:
            raise ValueError({'backbone_sha256': model.config.backbone_sha256})
    if payload['packed']:
        model.eval()
    model.load_weights(str(directory / 'adapter.safetensors'), strict=True)
    if training:
        model.train()
    elif not payload['packed']:
        model = model.snapshot()
    mx.eval(model.parameters())
    return (model, payload)

def import_torch_checkpoint(source, destination):
    """One-time CPU importer; does not load a Qwen backbone or run TorchAO."""
    import numpy as np
    import torch
    from transformers import AutoConfig
    saved = torch.load(source, map_location='cpu', weights_only=True)
    if saved.get('policy_kind') != 'qwen_ur_pose_v1_unfrozen' or saved.get('unfrozen_layers') != 2:
        raise ValueError({'policy_kind': saved.get('policy_kind'), 'unfrozen_layers': saved.get('unfrozen_layers')})
    adapter, old = (saved['adapter'], saved['model_config'])
    backbone_dtype = old.get('backbone_dtype', 'float32')
    if backbone_dtype not in ('float32', 'float16', 'bfloat16'):
        raise ValueError({'backbone_dtype': backbone_dtype})
    if (old['horizon'], old['execute_steps'], old.get('history_length', 0)) != (130, 130, 0):
        raise ValueError({'model_config': old})
    backbone = {name.removeprefix('encoder.backbone.'): mx.array(value.detach().float().cpu().numpy()) for name, value in adapter.items() if name.startswith('encoder.backbone.')}
    upstream = AutoConfig.from_pretrained(old['base_model'], revision=old['base_revision'])
    text = upstream.text_config
    if upstream.model_type != 'qwen3_5' or len(adapter['feature_norm.weight']) != text.hidden_size:
        raise ValueError({'model_type': upstream.model_type, 'feature_dim': len(adapter['feature_norm.weight'])})
    expected = {f'language_model.layers.{index}.mlp.{projection}.weight' for index in range(text.num_hidden_layers - 2, text.num_hidden_layers) for projection in ('gate_proj', 'up_proj', 'down_proj')}
    if set(backbone) != expected or not all((bool(mx.all(mx.isfinite(value))) for value in backbone.values())):
        raise ValueError({'missing_backbone_weights': sorted(expected - set(backbone))})
    for name, value in backbone.items():
        shape = (text.hidden_size, text.intermediate_size) if '.down_proj.' in name else (text.intermediate_size, text.hidden_size)
        if value.shape != shape:
            raise ValueError({'weight': name, 'shape': value.shape, 'expected': shape})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    backbone_path = destination / 'qwen_mlp.safetensors'
    mx.save_safetensors(str(backbone_path), backbone)
    config = PolicyConfig(feature_dim=len(adapter['feature_norm.weight']), base_model=old['base_model'], base_revision=old['base_revision'], horizon=old['horizon'], execute_steps=old['execute_steps'], history_length=old.get('history_length', 0), image_size=old['image_size'], workspace_low=tuple(old['workspace_low']), workspace_high=tuple(old['workspace_high']), backbone_sha256=file_hash(backbone_path))
    model = MacPolicy(config)
    model._backbone_file = backbone_path
    prefixes = {'state_projection': 'state_encoder.0', 'fusion_0': 'action_head.0', 'fusion_1': 'action_head.2', 'pose_projection': 'action_head.4'}
    mapped = {}
    for target, origin in prefixes.items():
        mapped[target + '.core.weight'] = origin + '.weight'
        mapped[target + '.bias'] = origin + '.bias'
    for name in ('feature_norm',):
        origin = name
        mapped[name + '.weight'], mapped[name + '.bias'] = (origin + '.weight', origin + '.bias')
    weights = []
    for target, origin in mapped.items():
        value = adapter[origin]
        weights.append((target, mx.array(value.detach().float().cpu().numpy())))
    model.load_weights(weights, strict=True)
    model._metadata = dict(source=str(Path(source).resolve()), source_sha256=file_hash(source), vocabulary=old['vocabulary'], source_quantization=saved.get('qat_scope'), fp8_training_completed=False, backbone_dtype=backbone_dtype)
    save_model(model, destination)
    return model

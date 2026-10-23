"""Check checkpoint persistence with a small encoder; no grasp claims."""

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import auto_unfreeze_qwen_pipeline as pipeline
import quick_eval_unfrozen
import train_ur_qwen_unfrozen
from metalvla_qat.qwen_pose_vla import QwenPoseVLA


def check_first_frame_loaders(directory):
    import numpy as np

    directory.mkdir()
    rows, first_frames = [], []
    for index in range(2):
        images = np.arange(6*3*4*6*3, dtype=np.uint8).reshape(6, 3, 4, 6, 3)
        images = images + index
        first_frames.append(images[0].copy())
        path = directory / f"{index}.npz"
        np.savez_compressed(path, images=images, state=np.full((6, 8), index, np.float32),
            previous=np.zeros((6, 8), np.float32), desired=np.zeros((6, 5), np.float32),
            metadata=np.array(json.dumps(dict(schema=pipeline.SCHEMA, success=True))))
        rows.append(dict(file=path.name, sha256=pipeline.sha256_file(path)))
    original_load = np.load
    reports = []
    for trainer in (pipeline, train_ur_qwen_unfrozen):
        references = []

        class Archive:
            def __init__(self, *args, **kwargs):
                self.archive = original_load(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.archive.close()

            def __getitem__(self, key):
                value = self.archive[key]
                if key == "images":
                    references.append(weakref.ref(value))
                return value

        preprocess = trainer.preprocess_rgb

        def checked_preprocess(value, size):
            assert len(references) == len(rows)
            assert all(reference() is None for reference in references)
            np.testing.assert_array_equal(value.numpy(), np.stack(first_frames))
            return preprocess(value, size)

        model = SimpleNamespace(config=SimpleNamespace(horizon=4, image_size=192), targets=lambda value: value)
        with patch.object(trainer, "DATA", directory), patch.object(trainer.np, "load", Archive), \
             patch.object(trainer, "preprocess_rgb", side_effect=checked_preprocess):
            images, tokens, state, previous, target = trainer.load_batch(rows, "cpu", model, {"fixture": 1})
        expected = preprocess(torch.from_numpy(np.stack(first_frames)), 192)
        torch.testing.assert_close(images, expected, rtol=0, atol=0)
        assert state.shape == previous.shape == (2, 8)
        assert target.shape == (2, 4, 5) and tokens.shape == (2, 1)
        reports.append(dict(trainer=trainer.__name__, full_episode_arrays_released=True, input_parity=True))
    return reports


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 64, bias=False)
        self.up_proj = nn.Linear(32, 64, bias=False)
        self.down_proj = nn.Linear(64, 32, bias=False)

    def forward(self, value):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value))
                              * self.up_proj(value))


class SmallEncoder(nn.Module):
    def __init__(self, vocabulary, **kwargs):
        super().__init__()
        self.backbone = nn.Module()
        count = 32 if kwargs.get("base_model") == "Qwen/Qwen3.5-9B" else 24
        self.backbone.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=32, num_hidden_layers=count))
        self.backbone.language_model = nn.Module()
        self.backbone.language_model.layers = nn.ModuleList()
        for _ in range(count):
            layer = nn.Module()
            layer.mlp = SmallMLP()
            layer.attention = nn.Linear(32, 32, bias=False)
            self.backbone.language_model.layers.append(layer)

    def extract_features(self, images, tokens):
        value = images.mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 32)
        for layer in self.backbone.language_model.layers:
            value = value + layer.mlp(value)
        return value


def synthetic_batch(rows, device, model, vocabulary):
    count = len(rows)
    return (
        torch.ones(count, 3, 3, 4, 4, device=device),
        torch.ones(count, 1, dtype=torch.long, device=device),
        torch.zeros(count, 8, device=device),
        torch.zeros(count, 8, device=device),
        torch.zeros(count, model.config.horizon, 6, device=device),
    )


def check_roundtrip(output, device, unfreeze_layers, trainer=pipeline, nine_b=False):
    models = []
    initial = []

    def make_model(*args, **kwargs):
        # Simulate loading the same pinned pretrained backbone on every call.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            with patch("metalvla_qat.qwen_pose_vla.QwenVLA", SmallEncoder):
                model = QwenPoseVLA(*args, **kwargs)
        if not models:
            initial.append(copy.deepcopy(model.state_dict()))
        models.append(model)
        return model

    output.mkdir()
    with patch.object(pipeline, "QwenPoseVLA", side_effect=make_model), \
         patch.object(trainer, "manifest_rows", side_effect=lambda _, split: [{}] * (2592 if split == "train" else 288)), \
         patch.object(trainer, "load_batch", side_effect=synthetic_batch):
        with redirect_stdout(io.StringIO()):
            if trainer is pipeline:
                options = dict(base_model="Qwen/Qwen3.5-9B", base_revision="synthetic-9b",
                               backbone_dtype="bfloat16", eval_batch_size=1, val_samples=3) if nine_b else {}
                pipeline.train_unfrozen_qwen(output, 2, 2, 1e-3, unfreeze_layers, device, 7, **options)
            else:
                arguments = [str(Path(trainer.__file__)), "--output-dir", str(output),
                             "--steps", "2", "--batch-size", "2", "--lr", "0.001",
                             "--unfreeze-layers", str(unfreeze_layers), "--device", device,
                             "--eval-every", "2", "--seed", "7"]
                if nine_b:
                    arguments.extend(["--base-model", "Qwen/Qwen3.5-9B", "--base-revision", "synthetic-9b",
                                      "--backbone-dtype", "bfloat16", "--eval-batch-size", "1", "--val-samples", "3"])
                with patch.object(trainer, "QwenPoseVLA", side_effect=make_model), \
                     patch.object(sys, "argv", arguments):
                    trainer.main()
                training = json.loads((output / "training.json").read_text())
                assert training["config"]["output_dir"] == str(output)
                assert training["training_log"][-1]["step"] == 2
        source = models[0].cpu().eval()
        prefixes = pipeline.unfrozen_backbone_prefixes(source, unfreeze_layers)
        state = source.state_dict()
        selected = {name for name in state if name.startswith(prefixes)}
        changed = {name for name in state if not torch.equal(state[name], initial[0][name])}
        assert selected <= changed
        assert not any(name.startswith("encoder.backbone.") and name not in selected
                       for name in changed)
        inputs = synthetic_batch([{}], "cpu", source, source.vocabulary)[:4]
        with torch.no_grad():
            expected = source(*inputs)[0]

        for filename in ("model_best.pt", "model_final.pt"):
            checkpoint = output / filename
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            saved_backbone = {name for name in saved["adapter"] if name.startswith("encoder.backbone.")}
            assert saved_backbone == selected
            assert saved["model_config"]["backbone_dtype"] == ("bfloat16" if nine_b else "float32")
            assert all(value.device.type == "cpu" for value in saved["adapter"].values())
            for loader in (pipeline.load_unfrozen_model,):
                restored, _ = loader(checkpoint, "cpu")
                for name, value in state.items():
                    torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(expected, restored(*inputs)[0], rtol=0, atol=0)

        # Mutating the live model after saving cannot change the file snapshot.
        with torch.no_grad():
            source.action_head[0].weight.add_(1)
        snapshot, _ = pipeline.load_unfrozen_model(output / "model_best.pt", "cpu")
        with torch.no_grad():
            torch.testing.assert_close(expected, snapshot(*inputs)[0], rtol=0, atol=0)

        invalid_cases = {}
        if selected:
            missing = copy.deepcopy(saved)
            for name in selected:
                del missing["adapter"][name]
            invalid_cases["missing_backbone"] = missing
            partial = copy.deepcopy(saved)
            del partial["adapter"][next(iter(selected))]
            invalid_cases["partial_backbone"] = partial
        task_missing = copy.deepcopy(saved)
        del task_missing["adapter"]["action_head.0.weight"]
        invalid_cases["missing_task_head"] = task_missing
        unknown = copy.deepcopy(saved)
        unknown["adapter"]["unknown.weight"] = torch.zeros(1)
        invalid_cases["unexpected_key"] = unknown
        for label, payload in invalid_cases.items():
            path = output / f"{label}.pt"
            torch.save(payload, path)
            try:
                pipeline.load_unfrozen_model(path, "cpu")
            except RuntimeError:
                pass
            else:
                raise AssertionError(label)

        # Validate both decoder layouts and runtime dtype without loading a real Qwen.
        from metalvla_qat.mac.model import import_torch_checkpoint, load_model
        from metalvla_qat.mac.cli import encoder_for
        from metalvla_qat.mac.checks import torch_float_reference
        import mlx.core as mx
        import numpy as np
        upstream = SimpleNamespace(model_type="qwen3_5", text_config=SimpleNamespace(
            hidden_size=32, intermediate_size=64, num_hidden_layers=32 if nine_b else 24))
        with patch("transformers.AutoConfig.from_pretrained", return_value=upstream):
            imported = import_torch_checkpoint(output / "model_best.pt", output / "mlx")
            bad = copy.deepcopy(saved)
            name = next(iter(selected))
            bad["adapter"][name] = bad["adapter"][name][:1]
            torch.save(bad, output / "invalid_shape.pt")
            try:
                import_torch_checkpoint(output / "invalid_shape.pt", output / "invalid_mlx")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid_mlp_shape")
        restored_mlx, _ = load_model(output / "mlx")
        with patch("metalvla_qat.mac.cli.OfficialQwenEncoder") as encoder:
            encoder_for(restored_mlx, device=device)
            assert encoder.call_args.args[0].dtype == ("bfloat16" if nine_b else "float32")
        features = np.ones((1,32), np.float32)
        states = np.zeros((1,8), np.float32)
        expected = torch_float_reference(torch.load(output / "model_best.pt", weights_only=True), features, states, states)
        actual = imported.actions(mx.array(features), mx.array(states), mx.array(states))
        mx.eval(actual)
        parity = float(np.max(np.abs(expected-np.asarray(actual))))
        assert parity < 1e-5

    return dict(trainer=trainer.__name__, device=device, unfreeze_layers=unfreeze_layers,
                persisted_backbone_tensors=len(selected), best_and_final_reload_max_error=0,
                rejected_invalid_cases=sorted(invalid_cases), nine_b_layout=nine_b,
                mlx_head_parity_error=parity)


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="metalvla-checkpoint-") as directory:
        root = Path(directory)
        loader_reports = check_first_frame_loaders(root / "loaders")
        cases = [("cpu", 2)]
        if torch.backends.mps.is_available():
            cases.append(("mps", 2))
        reports = [check_roundtrip(root / f"{device}-{count}", device, count)
                   for device, count in cases]
        standalone_cases = [("cpu", 2)]
        if torch.backends.mps.is_available():
            standalone_cases.append(("mps", 2))
        reports.extend(check_roundtrip(root / f"standalone-{device}-{count}", device, count,
                                       trainer=train_ur_qwen_unfrozen)
                       for device, count in standalone_cases)
        reports.extend(check_roundtrip(root / f"9b-{trainer.__name__}", "cpu", 2, trainer, nine_b=True)
                       for trainer in (pipeline, train_ur_qwen_unfrozen))
    print(json.dumps(dict(checkpoint_roundtrips=reports, first_frame_loaders=loader_reports, grasp_evaluation=False), indent=2))


if __name__ == "__main__":
    main()

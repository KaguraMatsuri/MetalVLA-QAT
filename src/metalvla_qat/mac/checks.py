"""Focused H130 Qwen/MLX persistence, QAT, RL and execution checks."""
from pathlib import Path
import tempfile
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from .data import build_cache, file_hash, load_cache, sample, write_json
from .encoder import EncoderConfig
from .model import MacPolicy, PolicyConfig, import_torch_checkpoint, load_model, save_model
from .runtime import Environments, rollout_wave, sample_actions
from .train import bc_update, masked_loss, reinforce_update, restore_resume, save_resume
from ..score import REWARD_VERSION
from .distribution import SAMPLING_VERSION

def assert_close(a, b, tolerance=1e-5):
    delta = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    if not np.isfinite(delta) or delta > tolerance:
        raise AssertionError(f"Numerical difference {delta} > {tolerance}")
    return delta

def torch_float_reference(saved, features, state, previous, history=None, valid=None):
    import torch
    from torch.nn import functional as F
    weights = {name: value for name, value in saved['adapter'].items()}
    f, s, p = (torch.from_numpy(value.copy()) for value in (features, state, previous))
    normalized = F.layer_norm(f, (f.shape[-1],), weights['feature_norm.weight'], weights['feature_norm.bias'], 1e-05)
    robot = torch.cat((s, p), -1)
    state_input = F.pad(robot, (0, 16))
    encoded = F.relu(F.linear(state_input, weights['state_encoder.0.weight'], weights['state_encoder.0.bias']))
    x = torch.cat((normalized, encoded), -1)
    for index in (0, 2):
        x = F.relu(F.linear(x, weights[f'action_head.{index}.weight'], weights[f'action_head.{index}.bias']))
    raw = F.linear(x, weights['action_head.4.weight'], weights['action_head.4.bias']).reshape(-1, saved['model_config']['horizon'], 6)
    return torch.cat((raw[..., :3].tanh(), raw[..., 3:5], raw[..., 5:6].tanh()), -1).detach().numpy()

class SyntheticEncoder:
    config = EncoderConfig()
    hidden_size = 64

    def encode(self, raw, instruction):
        return np.zeros((len(raw), 64), np.float32)

def run_checks(output, root, simulation=False):
    output, root = Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(synthetic=True, task_success_claimed=False)
    mx.random.seed(20261009)
    config = PolicyConfig(feature_dim=64)
    model = MacPolicy(config)
    f, s, p = mx.random.normal((2, 64)), mx.random.normal((2, 8)), mx.random.normal((2, 8))
    h, v = model.empty_history(2)
    assert h.shape == (2, 0, 80) and v.shape == (2, 0)
    model.enable_qat()
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.)
    target = mx.zeros((2, 130, 6))
    mask = mx.ones((2, 130), dtype=mx.bool_)
    batch = (f, s, p, h, v, target, mask)
    _, gradients = nn.value_and_grad(model, masked_loss)(model, *batch)
    mx.eval(gradients)
    report["quantized_weight_gradients"] = {
        name: float(mx.abs(gradients[name]["core"]["weight"]).max()) for name in model.linear_names}
    assert all(value > 0 for value in report["quantized_weight_gradients"].values())
    for _ in range(3):
        report["bc"] = bc_update(model, optimizer, batch)
    master = {name: np.asarray(value).copy() for name, value in tree_flatten(model.parameters())}
    snapshot = model.snapshot()
    report["packed_parity_error"] = assert_close(model.actions(f, s, p), snapshot.actions(f, s, p), .01)
    assert all(np.array_equal(np.asarray(value), master[name]) for name, value in tree_flatten(model.parameters()))
    save_model(model, output / "packed", packed=True)
    restored, _ = load_model(output / "packed")
    report["reload_error"] = assert_close(snapshot.actions(f, s, p), restored.actions(f, s, p), 0.)
    save_resume(model, optimizer, output / "resume", dict(step=3))
    resumed, _ = restore_resume(output / "resume", optim.AdamW(learning_rate=1e-3))
    report["resume_error"] = assert_close(model.actions(f, s, p), resumed.actions(f, s, p), 0.)
    report["qat_layers"] = list(model.linear_names)
    assert len(model.linear_names) == 4
    experiences=[]
    rng=np.random.default_rng(7)
    for index in range(2):
        inputs=[np.asarray(value)[index:index+1].copy() for value in (f,s,p,h,v)]
        reference=snapshot.actions(*(mx.array(value) for value in inputs))
        mx.eval(reference)
        reference=np.asarray(reference)
        _, latent, closed=sample_actions(reference,rng)
        decision=dict(zip(("features","state","previous","history","valid"),inputs))
        decision.update(reference=reference,latent=latent,closed=closed,executed_steps=130)
        experiences.append(dict(result=dict(score=float(index*8), training_reward=float(index*8), reward_version=REWARD_VERSION,
                                            sampling_version=SAMPLING_VERSION, success=bool(index)),decisions=[decision]))
    report["rl"]=reinforce_update(model,optimizer,experiences)
    assert report["rl"]["update_applied"]
    report["rl"]["synthetic_success_labels"]=report["rl"].pop("successes")
    with tempfile.TemporaryDirectory() as temporary:
        dataset=Path(temporary)/"dataset"
        raw=dataset/"raw/successes"
        raw.mkdir(parents=True)
        rows=[]
        for index,split in enumerate(("train","val")):
            frames=131
            images=np.stack([np.full((3,8,12,3), index*100+t,np.uint8) for t in range(frames)])
            desired=np.tile(np.array([0.,.1,.8,0.,0.],np.float32),(frames,1))
            state=np.full((frames,8),index,np.float32)
            path=raw/f"e{index}.npz"
            np.savez(path,images=images,desired=desired,state=state,previous=state)
            rows.append(dict(file=str(path.relative_to(dataset)),split=split,schema="ur5e_pick_place_pose_v1",success=True,sha256=file_hash(path),region=index,ordinal=0))
        (dataset/"raw/manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        class CacheEncoder(SyntheticEncoder):
            def prepare(self, raw, instruction): return raw
            def encode_prepared(self, raw): return np.repeat(raw[:,0,0,0,0,None].astype(np.float32),64,axis=1)
            def signature(self): return dict(sha256="synthetic_fixture")
        cache=Path(temporary)/"cache"
        build_cache(dataset,cache,CacheEncoder(),config,batch_size=4)
        episodes,_=load_cache(cache,"train",config)
        assert episodes[0]["features"][:,0].tolist()==[0.,130.]
        assert sample(episodes[0],1,config)[3].shape==(0,80)
        assert int(episodes[0]["valid"][1].sum())==1
        report["h130_cache_and_partial_chunk"] = True
    source=root/"data/ur_qwen_unfrozen_auto_20261009_143934/model_best.pt"
    if source.exists():
        import torch
        saved=torch.load(source,map_location="cpu",weights_only=True)
        imported=import_torch_checkpoint(source,output/"imported")
        width=imported.config.feature_dim
        features=rng.normal(size=(2,width)).astype(np.float32)
        states=rng.normal(size=(2,8)).astype(np.float32)
        previous=rng.normal(size=(2,8)).astype(np.float32)
        expected=torch_float_reference(saved,features,states,previous)
        actual=imported.actions(mx.array(features),mx.array(states),mx.array(previous))
        report["real_checkpoint_head_parity_error"]=assert_close(expected,actual,1e-5)
        report["real_backbone_sha256"]=imported.config.backbone_sha256
        assert file_hash(output/"imported/qwen_mlp.safetensors")==imported.config.backbone_sha256
        save_model(imported,output/"imported_copy")
        copied,_=load_model(output/"imported_copy")
        assert copied.config.backbone_sha256==imported.config.backbone_sha256
        report["portable_backbone_copy"]=True
    if simulation:
        dummy=MacPolicy(config)
        for name,value in tree_flatten(dummy.parameters()):
            if ".core.weight" in name: dummy.load_weights([(name,mx.zeros_like(value))],strict=False)
        bias=np.zeros((130,6),np.float32)
        bias[:,4],bias[:,5]=1.,-1.
        dummy.pose_projection.bias=mx.array(bias.reshape(-1))
        environments=Environments(count=2)
        try:
            episodes=rollout_wave(environments,dummy,SyntheticEncoder(),[20261008,20261009],[0,1],max_decisions=2)
            assert len(episodes)==2 and all(len(episode["decisions"])==1 for episode in episodes)
            assert all(not episode["result"]["expert_takeover"] for episode in episodes)
            dummy.pose_projection.bias=mx.zeros_like(dummy.pose_projection.bias)
            rejected=rollout_wave(environments,dummy,SyntheticEncoder(),[20261010,20261011],[2,3],max_decisions=2)
            assert all(episode["result"]["failure_type"]=="invalid_policy_action" and episode["decisions"][0]["executed_steps"]==0 for episode in rejected)
            report["h130_spawned_execution_and_invalid_yaw"] = True
        finally:
            environments.close()
    write_json(output/"result.json",report)
    return report

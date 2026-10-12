import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = True

    def __init__(self, *, named_tensors=None, flattened_tensor=None, metadata=None):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self._flattened_tensor = ("flattened", tuple(name for name, _ in named_tensors))
            self._metadata = tuple(name for name, _ in named_tensors)
            return

        self._flattened_tensor = flattened_tensor
        self._metadata = metadata

    def get_flattened_tensor(self):
        return self._flattened_tensor

    def get_metadata(self):
        return self._metadata


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(value, output_str):
        assert output_str is True
        return value


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return f"ref-{len(self.calls)}"


class _FakeEngine:
    def __init__(self):
        self.update_weights_from_tensor = _FakeRemoteMethod()


@pytest.mark.parametrize("supports_session", [True, False])
def test_colocated_weight_update_session_spans_all_chunks(monkeypatch, supports_session):
    module, _ = _load_update_weight_module(monkeypatch)
    events = []

    def remote_method(name, result=None):
        def remote(**kwargs):
            events.append(name)
            return result

        return types.SimpleNamespace(remote=remote)

    engine = types.SimpleNamespace(
        pause_generation=remote_method("pause"),
        flush_cache=remote_method("flush"),
        begin_weight_update=remote_method("begin", supports_session),
        end_weight_update=remote_method("end"),
        continue_generation=remote_method("continue"),
    )
    module.ray.get = lambda refs: refs
    module.dist.barrier = lambda group=None: events.append("barrier")
    module.accelerator.empty_cache = lambda: None
    module.post_process_weights = lambda **kwargs: events.append(
        "legacy_pre" if kwargs["restore_weights_before_load"] else "legacy_post"
    )

    updater = module.UpdateWeightFromTensor.__new__(module.UpdateWeightFromTensor)
    updater.rank = 0
    updater.weight_version = 0
    updater.rollout_engines = [engine]
    updater.quantization_config = {"quant_method": "compressed-tensors"}
    updater.weights_getter = lambda: {}
    updater._non_expert_param_info_buckets = None
    updater._full_param_info_buckets = ()
    updater._expert_transfer_plan = []
    updater._hf_weight_iterator = types.SimpleNamespace(get_hf_weight_chunks=lambda *args, **kwargs: [[], []])
    updater._send_hf_params = lambda params: (events.append("chunk") or [], [])

    updater.update_weights()

    if supports_session:
        assert events == [
            "pause", "flush", "begin", "barrier", "chunk", "barrier", "chunk", "barrier", "barrier", "end", "continue", "barrier"
        ]
    else:
        assert events == [
            "pause",
            "flush",
            "begin",
            "legacy_pre",
            "barrier",
            "chunk",
            "barrier",
            "chunk",
            "barrier",
            "barrier",
            "legacy_post",
            "end",
            "continue",
            "barrier",
        ]


def _install_fake_deps(monkeypatch):
    dist_state = types.SimpleNamespace(rank=0, world_size=2, gathered=None, local_object=None)

    slime_pkg = types.ModuleType("slime")
    slime_pkg.__path__ = [str(REPO_ROOT / "slime")]
    slime_backends_pkg = types.ModuleType("slime.backends")
    slime_backends_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends")]
    megatron_utils_pkg = types.ModuleType("slime.backends.megatron_utils")
    megatron_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils")]
    update_weight_pkg = types.ModuleType("slime.backends.megatron_utils.update_weight")
    update_weight_pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight")]
    slime_utils_pkg = types.ModuleType("slime.utils")
    slime_utils_pkg.__path__ = [str(REPO_ROOT / "slime" / "utils")]
    accelerator_mod = types.ModuleType("slime.utils.accelerator")
    accelerator_mod.device = lambda: "cuda:0"
    accelerator_mod.current_device = lambda: "cuda:0"
    accelerator_mod.ipc_collect = lambda: None

    dist_mod = types.ModuleType("torch.distributed")

    def gather_object(obj, object_gather_list, dst, group):
        dist_state.local_object = obj
        if object_gather_list is not None:
            object_gather_list[:] = dist_state.gathered(obj)

    dist_mod.get_rank = lambda: dist_state.rank
    dist_mod.get_world_size = lambda group=None: dist_state.world_size
    dist_mod.gather_object = gather_object

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    torch_mod.uint8 = "uint8"
    torch_mod.distributed = dist_mod
    torch_mod.empty = lambda size, dtype, device: {"size": size, "dtype": dtype, "device": device}
    torch_mod.no_grad = lambda: (lambda fn: fn)
    torch_mod.cuda = types.SimpleNamespace(current_device=lambda: "cuda:0", ipc_collect=lambda: None)
    torch_mod.nn = types.SimpleNamespace(Module=object)

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod

    sglang_mod = types.ModuleType("slime.backends.megatron_utils.sglang")
    sglang_mod.FlattenedTensorBucket = _FakeFlattenedTensorBucket
    sglang_mod.MultiprocessingSerializer = _FakeMultiprocessingSerializer

    megatron_to_hf_mod = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf_mod.convert_to_hf = lambda *args, **kwargs: []

    expert_routing_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.expert_routing")
    expert_routing_mod.configure_expert_routing = lambda *args, **kwargs: (None, [])

    hf_weight_iterator_direct_mod = types.ModuleType(
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
    )
    hf_weight_iterator_direct_mod.HfWeightIteratorDirect = lambda *args, **kwargs: None

    slime_utils_types_mod = types.ModuleType("slime.utils.types")
    slime_utils_types_mod.ParamInfo = type("ParamInfo", (), {})

    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: object()

    update_from_distributed_mod = types.ModuleType(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    )
    update_from_distributed_mod.connect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.disconnect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.post_process_weights = lambda *args, **kwargs: None
    update_from_distributed_mod.update_weights_from_distributed = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "slime", slime_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends", slime_backends_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils", megatron_utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight", update_weight_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils", slime_utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils.accelerator", accelerator_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor_mod)
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.megatron_to_hf", megatron_to_hf_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight.expert_routing", expert_routing_mod)
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
        hf_weight_iterator_direct_mod,
    )
    monkeypatch.setitem(sys.modules, "slime.utils.types", slime_utils_types_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.distributed_utils", distributed_utils_mod)
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        update_from_distributed_mod,
    )

    return dist_state


def _load_update_weight_module(monkeypatch):
    dist_state = _install_fake_deps(monkeypatch)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_tensor"
    sys.modules.pop(module_name, None)
    module_path = (
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_tensor.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, dist_state


def test_v4_fused_weights_stay_in_one_update_bucket(monkeypatch):
    _load_update_weight_module(monkeypatch)
    mpu = sys.modules["megatron.core.mpu"]
    mpu.get_tensor_model_parallel_world_size = lambda: 1
    mpu.get_expert_tensor_parallel_world_size = lambda: 1
    sys.modules["slime.backends.megatron_utils.sglang"].monkey_patch_torch_reductions = lambda: None

    common = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.all_gather_params_async = lambda params: params
    common.named_params_and_buffers = lambda args, model: []
    monkeypatch.setitem(sys.modules, common.__name__, common)

    module_name = "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
    module_path = REPO_ROOT / "slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    names = [
        "module.module.decoder.layers.0.self_attention.linear_kv_proj.weight",
        "module.module.decoder.layers.0.self_attention.linear_q_down_proj.weight",
        "module.module.decoder.layers.0.self_attention.q_layernorm.weight",
        "module.module.decoder.layers.0.self_attention.core_attention.compressor.linear_wgate.weight",
        "module.module.decoder.layers.0.self_attention.core_attention.compressor.linear_wkv.weight",
    ]
    infos = [types.SimpleNamespace(name=name, size=6) for name in names]
    buckets = module.pack_param_info_buckets(infos, 10, keep_v4_fusion_pairs=True)
    bucket_names = [[info.name for info in bucket] for bucket in buckets]

    assert set(names[:2]) == set(bucket_names[0])
    assert set(names[3:]) == set(bucket_names[2])
    assert bucket_names[1] == [names[2]]

    common.all_gather_param = lambda name, param: param
    common.named_params_and_buffers = lambda args, model: [(name, types.SimpleNamespace()) for name in names]
    sys.modules["slime.utils.distributed_utils"].init_process_group = lambda *args, **kwargs: None
    sys.modules["slime.backends.megatron_utils.megatron_to_hf"].convert_to_hf = lambda args, model, name, param, quant: [
        (name, types.SimpleNamespace(numel=lambda: 6, element_size=lambda: 1))
    ]

    distributed_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    distributed_path = REPO_ROOT / "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py"
    distributed_spec = importlib.util.spec_from_file_location(distributed_name, distributed_path)
    distributed_module = importlib.util.module_from_spec(distributed_spec)
    monkeypatch.setitem(sys.modules, distributed_name, distributed_module)
    assert distributed_spec.loader is not None
    distributed_spec.loader.exec_module(distributed_module)

    updater = distributed_module.UpdateWeightFromDistributed.__new__(distributed_module.UpdateWeightFromDistributed)
    updater.args = types.SimpleNamespace(update_weight_buffer_size=10)
    updater.model = []
    updater.model_name = "DeepseekV4ForCausalLM"
    updater.quantization_config = None
    updater._is_pp_src_rank = True
    distributed_chunks = list(updater._iter_non_expert_chunks())
    assert [set(name for name, _ in chunk) for chunk in distributed_chunks] == [
        set(names[:2]),
        {names[2]},
        set(names[3:]),
    ]


@pytest.mark.parametrize("supports_session", [True, False])
def test_distributed_weight_update_session_spans_all_chunks(monkeypatch, supports_session):
    _load_update_weight_module(monkeypatch)
    sys.modules["slime.utils.distributed_utils"].init_process_group = lambda *args, **kwargs: None
    sys.modules["slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"].group_v4_fusion_pairs = (
        lambda items, get_name: ([item] for item in items)
    )
    common = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.all_gather_param = lambda name, param: param
    common.named_params_and_buffers = lambda args, model: []
    monkeypatch.setitem(sys.modules, common.__name__, common)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    module_path = REPO_ROOT / "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    events = []

    def remote_method(name, result=None):
        def remote(**kwargs):
            events.append(name)
            return result

        return types.SimpleNamespace(remote=remote)

    engine = types.SimpleNamespace(
        pause_generation=remote_method("pause"),
        flush_cache=remote_method("flush"),
        begin_weight_update=remote_method("begin", supports_session),
        end_weight_update=remote_method("end"),
        continue_generation=remote_method("continue"),
    )
    module.ray.get = lambda refs: refs
    module.dist.barrier = lambda group=None: events.append("barrier")
    module.post_process_weights = lambda **kwargs: events.append(
        "legacy_pre" if kwargs["restore_weights_before_load"] else "legacy_post"
    )

    updater = module.UpdateWeightFromDistributed.__new__(module.UpdateWeightFromDistributed)
    updater.weight_version = 0
    updater.rollout_engines = [engine]
    updater.quantization_config = {"quant_method": "compressed-tensors"}
    updater._is_pp_src_rank = False
    updater._send_weights = lambda pbar: events.append("chunks")
    updater.update_weights()

    assert events == (
        ["pause", "flush", "begin", "barrier", "chunks", "end", "continue", "barrier"]
        if supports_session
        else [
            "pause", "flush", "begin", "legacy_pre", "barrier", "chunks", "legacy_post", "end", "continue", "barrier"
        ]
    )


def test_empty_colocated_bucket_still_participates_in_gather(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    dist_state.gathered = lambda local: [local, []]
    engine = _FakeEngine()

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=3,
    )

    assert dist_state.local_object == []
    assert refs == []
    assert long_lived_tensors == []
    assert engine.update_weights_from_tensor.calls == []


def test_source_rank_pads_empty_colocated_bucket_entries(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    remote_serialized_bucket = {"flattened_tensor": ("remote",), "metadata": ("remote_weight",)}
    dist_state.gathered = lambda local: [local, [remote_serialized_bucket]]
    engine = _FakeEngine()

    refs, long_lived_tensors = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=7,
    )

    assert refs == ["ref-1"]
    assert len(long_lived_tensors) == 1
    empty_bucket = long_lived_tensors[0]
    assert empty_bucket["metadata"] == []
    assert empty_bucket["flattened_tensor"] == {"size": 0, "dtype": "uint8", "device": "cuda:0"}

    assert engine.update_weights_from_tensor.calls == [
        {
            "serialized_named_tensors": [empty_bucket, remote_serialized_bucket],
            "load_format": "flattened_bucket",
            "weight_version": "7",
        }
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

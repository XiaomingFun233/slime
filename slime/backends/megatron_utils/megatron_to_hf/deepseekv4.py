"""Map Megatron dsv4_hybrid parameters to the names SGLang's V4 loader accepts.

Those names are the post-``remap_weight_name_to_dpsk_hf_format`` names
(``model.embed_tokens``, ``model.layers.*.self_attn.wq_a``, ``mlp.experts.*.gate_proj``),
not the on-disk release names (``embed``, ``layers.*.attn.wq_a``, ``ffn.experts.*.w1``).
"""

from __future__ import annotations

import re

import torch


def _ape_to_checkpoint(ape: torch.Tensor, ratio: int) -> torch.Tensor:
    """Undo the SGLang/Megatron ratio4 overlap layout for release weights."""
    if ratio != 4:
        return ape
    if ape.shape[0] != 4 or ape.shape[1] % 2:
        raise ValueError(f"DeepSeek-V4 ratio4 APE has invalid shape {tuple(ape.shape)}")
    parts = ape.reshape(4, 2, -1)
    first = parts[:2].reshape(4, -1)
    second = parts[2:].reshape(4, -1)
    return torch.cat((first, second), dim=1)


def _compress_ratio(args, layer: int) -> int | None:
    ratios = getattr(args, "csa_compress_ratios", None)
    if ratios is None:
        ratios = getattr(args, "compress_ratios", None)
    if ratios is None or layer >= len(ratios):
        return None
    return int(ratios[layer])


def _named_params(args):
    return getattr(args, "_slime_megatron_named_params", None)


def _hc_scale(alpha_pre: torch.Tensor, alpha_post: torch.Tensor, alpha_res: torch.Tensor) -> torch.Tensor:
    return torch.cat((alpha_pre.reshape(1), alpha_post.reshape(1), alpha_res.reshape(1)))


def convert_deepseekv4_to_hf(args, name, param):
    direct = {
        "module.module.embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "module.module.output_layer.weight": "lm_head.weight",
        "module.module.decoder.final_layernorm.weight": "model.norm.weight",
        "module.module.decoder.hc_head_fn": "model.hc_head_fn",
        "module.module.decoder.hc_head_base": "model.hc_head_base",
        "module.module.decoder.hc_head_scale": "model.hc_head_scale",
    }
    if name in direct:
        return [(direct[name], param)]

    scale_match = re.fullmatch(
        r"module\.module\.(decoder|mtp)\.layers\.(\d+)\."
        r"(?:transformer_layer\.)?"
        r"(self_attention_hyper_connection|mlp_hyper_connection)\.(alpha_pre|alpha_post|alpha_res)",
        name,
    )
    if scale_match:
        block, layer_idx, which, attr = scale_match.groups()
        if attr != "alpha_pre":
            return []
        params = _named_params(args)
        if params is None:
            raise ValueError(
                "DeepSeek-V4 mHC export needs the sibling alpha_pre/alpha_post/alpha_res "
                "parameters from the same layer"
            )
        layer_prefix = f"module.module.{block}.layers.{layer_idx}."
        if block == "mtp" and ".transformer_layer." in name:
            layer_prefix += "transformer_layer."
        prefix = layer_prefix + f"{which}."
        parts = {key: params.get(prefix + key) for key in ("alpha_pre", "alpha_post", "alpha_res")}
        if any(value is None for value in parts.values()):
            raise ValueError(f"Missing DeepSeek-V4 mHC scale siblings for {prefix}")
        hf_layer = int(layer_idx) if block == "decoder" else int(layer_idx) + args.num_layers
        hc = "hc_attn" if which == "self_attention_hyper_connection" else "hc_ffn"
        return [(f"model.layers.{hf_layer}.{hc}_scale", _hc_scale(parts["alpha_pre"], parts["alpha_post"], parts["alpha_res"]))]

    layer_match = re.fullmatch(r"module\.module\.(decoder|mtp)\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise ValueError(f"Unknown DeepSeek-V4 parameter name: {name}")
    block, layer_idx, rest = layer_match.groups()
    if block == "mtp":
        rest = rest.removeprefix("transformer_layer.")
        hf_layer = int(layer_idx) + args.num_layers
        mtp_direct = {
            "eh_proj.weight": "eh_proj.weight",
            "enorm.weight": "enorm.weight",
            "hnorm.weight": "hnorm.weight",
            "final_layernorm.weight": "shared_head.norm.weight",
        }
        if rest in mtp_direct:
            return [(f"model.layers.{hf_layer}.{mtp_direct[rest]}", param)]
    else:
        hf_layer = int(layer_idx)
    return _layer_to_hf(args, hf_layer, rest, param)


def _layer_to_hf(args, layer_idx: int, rest: str, param: torch.Tensor):
    prefix = f"model.layers.{layer_idx}"
    direct = {
        "input_layernorm.weight": "input_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "self_attention.linear_q_down_proj.weight": "self_attn.wq_a.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.linear_q_up_proj.weight": "self_attn.wq_b.weight",
        "self_attention.linear_kv_proj.weight": "self_attn.wkv.weight",
        "self_attention.kv_layernorm.weight": "self_attn.kv_norm.weight",
        "self_attention.linear_o_group_proj": "self_attn.wo_a.weight",
        "self_attention.linear_proj.weight": "self_attn.wo_b.weight",
        "self_attention.core_attention.attn_sink": "self_attn.attn_sink",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        "mlp.router.tid2eid": "mlp.topk.tid2eid",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
        "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.down_proj.weight",
    }
    if rest in direct:
        return [(f"{prefix}.{direct[rest]}", param)]

    if rest == "mlp.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [
            (f"{prefix}.mlp.gate_proj.weight", gate),
            (f"{prefix}.mlp.up_proj.weight", up),
        ]
    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [
            (f"{prefix}.mlp.shared_experts.gate_proj.weight", gate),
            (f"{prefix}.mlp.shared_experts.up_proj.weight", up),
        ]

    hyper = re.fullmatch(
        r"(self_attention_hyper_connection|mlp_hyper_connection)\.(mapping_proj\.weight|bias)",
        rest,
    )
    if hyper:
        which, attr = hyper.groups()
        hc = "hc_attn" if which == "self_attention_hyper_connection" else "hc_ffn"
        suffix = "fn" if attr == "mapping_proj.weight" else "base"
        return [(f"{prefix}.{hc}_{suffix}", param)]

    compressor = re.fullmatch(
        r"self_attention\.core_attention\.(indexer\.)?compressor\.(linear_wkv\.weight|linear_wgate\.weight|norm\.weight|ape)",
        rest,
    )
    if compressor:
        nested, attr = compressor.groups()
        hf_attr = {
            "linear_wkv.weight": "wkv.weight",
            "linear_wgate.weight": "wgate.weight",
            "norm.weight": "norm.weight",
            "ape": "ape",
        }[attr]
        head = "self_attn.indexer.compressor" if nested else "self_attn.compressor"
        if hf_attr == "ape":
            param = _ape_to_checkpoint(param, _compress_ratio(args, layer_idx) or 0)
        return [(f"{prefix}.{head}.{hf_attr}", param)]

    indexer = re.fullmatch(r"self_attention\.core_attention\.indexer\.(linear_wq_b|linear_weights_proj)\.weight", rest)
    if indexer:
        hf_name = {"linear_wq_b": "wq_b", "linear_weights_proj": "weights_proj"}[indexer.group(1)]
        return [(f"{prefix}.self_attn.indexer.{hf_name}.weight", param)]

    expert = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if expert:
        projection, expert_id = expert.groups()
        if projection == "1":
            gate, up = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight", gate),
                (f"{prefix}.mlp.experts.{expert_id}.up_proj.weight", up),
            ]
        return [(f"{prefix}.mlp.experts.{expert_id}.down_proj.weight", param)]

    raise ValueError(f"Unknown DeepSeek-V4 parameter name: {rest}")

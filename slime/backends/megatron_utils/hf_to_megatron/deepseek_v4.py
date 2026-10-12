"""Map a DeepSeek-V4 release checkpoint onto Megatron's dsv4_hybrid modules.

SGLang's loader remaps the release names (``embed.*``, ``layers.*.attn.*``,
``layers.*.ffn.*``, ``hc_*``) onto the module names below before copying.
This loader accepts both those release names and the post-remap names emitted
by ``convert_deepseekv4_to_hf`` for live weight updates.
"""

from __future__ import annotations

import re

import torch

from .common import SafetensorReader, merge_gate_up, strip_mcore_wrappers


def _ape_from_checkpoint(ape: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert the release C4 APE layout to Megatron's overlap layout."""
    if ratio != 4:
        return ape
    if ape.shape[0] != 4 or ape.shape[1] % 2:
        raise ValueError(f"DeepSeek-V4 ratio4 APE has invalid shape {tuple(ape.shape)}")
    first, second = ape.chunk(2, dim=-1)
    return torch.cat((first, second), dim=0).reshape_as(ape)


def _compress_ratio(config, layer: int) -> int | None:
    ratios = getattr(config, "compress_ratios", None)
    if ratios is None:
        ratios = getattr(config, "csa_compress_ratios", None)
    if ratios is None:
        layer_types = getattr(config, "layer_types", None)
        rates = getattr(config, "compress_rates", None)
        if layer_types is not None and layer < len(layer_types):
            if not isinstance(rates, dict):
                rates = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
            ratios = [rates.get(layer_type, 0) for layer_type in layer_types]
    if ratios is None or layer >= len(ratios):
        return None
    return int(ratios[layer])


class _DeepSeekV4Reader:
    """Read both SGLang-remapped names and official release names."""

    def __init__(self, reader: SafetensorReader, config):
        self.reader = reader
        self.config = config

    def _official_name(self, name: str) -> str:
        if name == "model.embed_tokens.weight":
            return "embed.weight"
        if name == "lm_head.weight":
            return "head.weight"
        if name == "model.norm.weight":
            return "norm.weight"
        if name.startswith("model.hc_head_"):
            return name.removeprefix("model.")
        layer_match = re.match(r"model\.layers\.(\d+)\.(.+)", name)
        if layer_match and int(layer_match.group(1)) >= self.config.num_hidden_layers:
            mtp_layer = int(layer_match.group(1)) - self.config.num_hidden_layers
            name = f"mtp.{mtp_layer}.{layer_match.group(2)}"
            name = name.replace(".shared_head.norm.weight", ".norm.weight")
        else:
            name = name.replace("model.layers.", "layers.")
        name = name.replace(".self_attn.", ".attn.")
        name = name.replace(".mlp.", ".ffn.")
        name = name.replace(".input_layernorm.", ".attn_norm.")
        name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
        name = name.replace(".gate_proj.", ".w1.")
        name = name.replace(".down_proj.", ".w2.")
        name = name.replace(".up_proj.", ".w3.")
        name = name.replace(".topk.tid2eid", ".gate.tid2eid")
        name = name.replace(".gate.e_score_correction_bias", ".gate.bias")
        return name

    def __contains__(self, name: str) -> bool:
        return name in self.reader or self._official_name(name) in self.reader

    def get_tensor(self, name: str, *, layer: int | None = None) -> torch.Tensor:
        source_name = name if name in self.reader else self._official_name(name)
        tensor = self.reader.get_tensor(source_name)
        if layer is None:
            match = re.search(r"(?:model\.)?layers\.(\d+)", name)
            layer = int(match.group(1)) if match else None
        if layer is not None and ".compressor.ape" in name:
            tensor = _ape_from_checkpoint(tensor, _compress_ratio(self.config, layer) or 0)
        return tensor

_DIRECT = {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "decoder.final_layernorm.weight": "model.norm.weight",
    "decoder.hc_head_fn": "model.hc_head_fn",
    "decoder.hc_head_base": "model.hc_head_base",
    "decoder.hc_head_scale": "model.hc_head_scale",
}


def _output_layer(reader: SafetensorReader, config) -> torch.Tensor:
    if getattr(config, "tie_word_embeddings", False) or "lm_head.weight" not in reader:
        return reader.get_tensor("model.embed_tokens.weight")
    return reader.get_tensor("lm_head.weight")


def _hc_alphas(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Megatron broadcasts three scalars; SGLang stores the same three scales.
    return scale[0:1], scale[1:2], scale[2:3]


def _layer_tensor(layer: int, rest: str, reader: SafetensorReader) -> torch.Tensor:
    rest = rest.removeprefix("mtp_model_layer.")
    prefix = f"model.layers.{layer}"
    direct = {
        "hc_head_fn": "hc_head_fn",
        "hc_head_base": "hc_head_base",
        "hc_head_scale": "hc_head_scale",
        "e_proj.weight": "e_proj.weight",
        "h_proj.weight": "h_proj.weight",
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
        "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.down_proj.weight",
    }
    if rest in direct:
        return reader.get_tensor(f"{prefix}.{direct[rest]}")

    hyper = re.fullmatch(
        r"(self_attention_hyper_connection|mlp_hyper_connection)\.(mapping_proj\.weight|alpha_pre|alpha_post|alpha_res|bias)",
        rest,
    )
    if hyper:
        which, attr = hyper.groups()
        hc = "hc_attn" if which == "self_attention_hyper_connection" else "hc_ffn"
        if attr == "mapping_proj.weight":
            return reader.get_tensor(f"{prefix}.{hc}_fn")
        if attr == "bias":
            return reader.get_tensor(f"{prefix}.{hc}_base")
        scale = reader.get_tensor(f"{prefix}.{hc}_scale")
        alpha_pre, alpha_post, alpha_res = _hc_alphas(scale)
        return {"alpha_pre": alpha_pre, "alpha_post": alpha_post, "alpha_res": alpha_res}[attr]

    compressor = re.fullmatch(
        r"self_attention\.core_attention\.(?:indexer\.)?compressor\.(linear_wkv\.weight|linear_wgate\.weight|norm\.weight|ape)",
        rest,
    )
    if compressor:
        hf_rest = {
            "linear_wkv.weight": "wkv.weight",
            "linear_wgate.weight": "wgate.weight",
            "norm.weight": "norm.weight",
            "ape": "ape",
        }[compressor.group(1)]
        attn_rest = rest.split("compressor.", 1)[0] + "compressor." + hf_rest
        attn_rest = attn_rest.removeprefix("self_attention.core_attention.")
        # indexer.compressor.* stays nested; the main compressor is self_attn.compressor.*.
        if attn_rest.startswith("indexer."):
            return reader.get_tensor(f"{prefix}.self_attn.{attn_rest}")
        return reader.get_tensor(f"{prefix}.self_attn.compressor.{hf_rest}")

    indexer = re.fullmatch(r"self_attention\.core_attention\.indexer\.(linear_wq_b|linear_weights_proj)\.weight", rest)
    if indexer:
        hf_name = {"linear_wq_b": "wq_b", "linear_weights_proj": "weights_proj"}[indexer.group(1)]
        return reader.get_tensor(f"{prefix}.self_attn.indexer.{hf_name}.weight")

    if rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    if rest == "mlp.shared_experts.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.shared_experts.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.shared_experts.up_proj.weight"),
        )
    expert = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if expert:
        projection, expert_id = expert.groups()
        if projection == "1":
            return merge_gate_up(
                reader.get_tensor(f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight"),
                reader.get_tensor(f"{prefix}.mlp.experts.{expert_id}.up_proj.weight"),
            )
        return reader.get_tensor(f"{prefix}.mlp.experts.{expert_id}.down_proj.weight")
    raise KeyError(f"Unsupported DeepSeek-V4 Megatron layer parameter {rest!r}")


def deepseek_v4_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    reader = _DeepSeekV4Reader(reader, config)
    name = strip_mcore_wrappers(name)
    if name == "output_layer.weight":
        return _output_layer(reader, config)
    if name in _DIRECT:
        return reader.get_tensor(_DIRECT[name])

    mtp = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp:
        mtp_layer, rest = mtp.groups()
        layer = config.num_hidden_layers + int(mtp_layer)
        direct = {
            "eh_proj.weight": "eh_proj.weight",
            "e_proj.weight": "e_proj.weight",
            "h_proj.weight": "h_proj.weight",
            "enorm.weight": "enorm.weight",
            "hnorm.weight": "hnorm.weight",
            "final_layernorm.weight": "shared_head.norm.weight",
            "hc_head_fn": "hc_head_fn",
            "hc_head_base": "hc_head_base",
            "hc_head_scale": "hc_head_scale",
        }
        if rest in direct:
            return reader.get_tensor(f"model.layers.{layer}.{direct[rest]}")
        rest = rest.removeprefix("transformer_layer.").removeprefix("mtp_model_layer.")
        return _layer_tensor(layer, rest, reader)

    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported DeepSeek-V4 Megatron parameter {name!r}")
    return _layer_tensor(int(match.group(1)), match.group(2), reader)

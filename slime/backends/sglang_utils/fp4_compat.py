"""Compatibility policy for online DeepSeek-V4-Flash weight updates."""

import json
from pathlib import Path


def needs_dsv4_fp4_dequant(model_path: str | None, hf_checkpoint: str | None) -> bool:
    if not model_path or not hf_checkpoint:
        return False
    model_dir = Path(model_path).resolve()
    if model_dir != Path(hf_checkpoint).resolve():
        return False
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open() as config_file:
        config = json.load(config_file)
    return config.get("model_type") == "deepseek_v4" and config.get("expert_dtype") == "fp4"

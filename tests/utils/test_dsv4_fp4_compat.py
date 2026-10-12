import json

import pytest

from slime.backends.sglang_utils.fp4_compat import needs_dsv4_fp4_dequant


@pytest.mark.parametrize(
    ("model_type", "expert_dtype", "expected"),
    [
        ("deepseek_v4", "fp4", True),
        ("deepseek_v4", None, False),
        ("deepseek_v3", "fp4", False),
    ],
)
def test_dsv4_fp4_dequant_only_for_matching_checkpoint(tmp_path, model_type, expert_dtype, expected):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": model_type, "expert_dtype": expert_dtype}))

    assert needs_dsv4_fp4_dequant(str(checkpoint), str(checkpoint)) is expected
    assert not needs_dsv4_fp4_dequant(str(checkpoint), str(tmp_path / "other"))
    assert not needs_dsv4_fp4_dequant(None, str(checkpoint))

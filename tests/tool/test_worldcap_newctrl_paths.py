import subprocess
from pathlib import Path


ROOT = Path("/home/zhaodanqi/clone/WoTE")
HELPER = ROOT / "tool/common/worldcap_newctrl_paths.sh"


def _source_and_dump(*names: str) -> dict[str, str]:
    joined = " ".join(names)
    cmd = (
        f"source '{HELPER}' >/dev/null 2>&1 && "
        f"for v in {joined}; do printf '%s=%s\\n' \"$v\" \"${{!v}}\"; done"
    )
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    result = {}
    for line in out.strip().splitlines():
        key, value = line.split("=", 1)
        result[key] = value
    return result


def test_helper_exports_newctrl_default_paths():
    values = _source_and_dump(
        "WORLDCAP_DATA_ROOT",
        "WORLDCAP_CTRL_REF_64",
        "WORLDCAP_CTRL_EXEC_64",
        "WORLDCAP_CTRL_REF_128",
        "WORLDCAP_CTRL_EXEC_128",
        "WORLDCAP_CTRL_REF_1024",
        "WORLDCAP_CTRL_EXEC_1024",
    )
    assert values["WORLDCAP_DATA_ROOT"] == "/home/zhaodanqi/clone/WoTE/newCtrl"
    assert values["WORLDCAP_CTRL_REF_64"].endswith("/newCtrl/controller/ref_trajs/Anchors_Original_64_centered.npy")
    assert values["WORLDCAP_CTRL_EXEC_64"].endswith("/newCtrl/controller/bundles/64/controller_styles_64.npz")
    assert values["WORLDCAP_CTRL_REF_128"].endswith("/newCtrl/controller/ref_trajs/Anchors_Original_128_centered.npy")
    assert values["WORLDCAP_CTRL_EXEC_128"].endswith("/newCtrl/controller/bundles/128/controller_styles_128.npz")
    assert values["WORLDCAP_CTRL_REF_1024"].endswith("/newCtrl/controller/ref_trajs/Anchors_Original_1024_centered.npy")
    assert values["WORLDCAP_CTRL_EXEC_1024"].endswith("/newCtrl/controller/bundles/1024/controller_styles_1024.npz")


def test_training_wrappers_source_helper_and_use_newctrl_defaults():
    script_paths = [
        ROOT / "tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl64.sh",
        ROOT / "tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128.sh",
        ROOT / "tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl1024.sh",
    ]
    for path in script_paths:
        text = path.read_text(encoding="utf-8")
        assert 'source "${ROOT}/tool/common/worldcap_newctrl_paths.sh"' in text
        assert "WORLDCAP_CTRL_REF_" in text
        assert "WORLDCAP_CTRL_EXEC_" in text


def test_eval_script_sources_helper_and_uses_newctrl_defaults():
    path = ROOT / "tool/evaluate/eval_ckpts_20260225_valstyles_v3_20260303_ref64_128_1024.sh"
    text = path.read_text(encoding="utf-8")
    assert 'source "${ROOT}/tool/common/worldcap_newctrl_paths.sh"' in text
    assert 'CTRL_REF_64=${CTRL_REF_64:-"${WORLDCAP_CTRL_REF_64}"}' in text
    assert 'CTRL_EXEC_64=${CTRL_EXEC_64:-"${WORLDCAP_CTRL_EXEC_64}"}' in text
    assert 'CTRL_REF_128=${CTRL_REF_128:-"${WORLDCAP_CTRL_REF_128}"}' in text
    assert 'CTRL_EXEC_128=${CTRL_EXEC_128:-"${WORLDCAP_CTRL_EXEC_128}"}' in text
    assert 'CTRL_REF_1024=${CTRL_REF_1024:-"${WORLDCAP_CTRL_REF_1024}"}' in text
    assert 'CTRL_EXEC_1024=${CTRL_EXEC_1024:-"${WORLDCAP_CTRL_EXEC_1024}"}' in text

#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-}
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: bash tool/training/train_from_yaml.sh path/to/config.yaml [extra hydra overrides...]" >&2
  exit 2
fi
shift || true

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERR] Config file does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

_generated=$(
  python - "${CONFIG_PATH}" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(config_path.read_text()) or {}


def get(path, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve(value):
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def repl(match):
            key = match.group(1)
            found = get(key)
            if found is None:
                return match.group(0)
            return str(resolve(found))

        return pattern.sub(repl, value)
    if isinstance(value, list):
        return [resolve(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve(val) for key, val in value.items()}
    return value


cfg = resolve(cfg)


def q(value):
    return shlex.quote(str(value))


def emit_export(name, value):
    if value is not None:
        print(f"export {name}={q(value)}")


run = cfg.get("run", {})
data = cfg.get("data", {})
model = cfg.get("model", {})
controller = model.get("controller", {})
switches = model.get("switches", {})
train = cfg.get("train", {})
freeze = train.get("freeze", {})
losses = train.get("losses", {})
optim = train.get("optimizer", {})

root = run.get("root", "/home/zhaodanqi/clone/WoTE")
python = run.get("python", "python")
gpu_ids = run.get("gpu_ids")
devices = run.get("devices")
exp_root = run.get("exp_root")
run_name = run.get("name", config_path.stem)
tags = run.get("tags", [])
if isinstance(tags, list):
    tags = ",".join(str(item) for item in tags)

emit_export("WOTE_YAML_CONFIG", config_path)
emit_export("ROOT", root)
emit_export("PYTHON", python)
emit_export("CUDA_VISIBLE_DEVICES", gpu_ids)
emit_export("NAVSIM_EXP_ROOT", exp_root)
emit_export("WOTE_CTRL_STYLE_SPLIT", controller.get("style_split", "train"))
emit_export("WOTE_WANDB_PROJECT", run.get("wandb_project"))
emit_export("WOTE_WANDB_RUN_NAME", run_name)
emit_export("WOTE_WANDB_TAGS", tags)
emit_export("WOTE_PRINT_TRAINABLE", "1" if freeze.get("print_trainable", False) else "0")
emit_export("WOTE_INIT_CKPT", model.get("init_ckpt"))

required_paths = {
    "root": root,
    "init_ckpt": model.get("init_ckpt"),
    "cache_path": data.get("cache_path"),
    "planner_anchors": data.get("planner_anchors"),
    "controller_style_ref": controller.get("style_ref_path", controller.get("ref_path")),
    "controller_style_exec": controller.get("style_exec_path", controller.get("exec_path")),
    "controller_candidate_exec": controller.get("candidate_exec_path", controller.get("exec_path")),
}
missing = [f"{key}={path}" for key, path in required_paths.items() if path and not Path(path).exists()]
if missing:
    print("echo '[ERR] Missing required path(s):' >&2")
    for item in missing:
        print(f"echo {q(item)} >&2")
    print("exit 3")
    sys.exit(0)


def bool_s(value):
    return "true" if bool(value) else "false"


def list_s(value):
    if value is None:
        value = []
    if isinstance(value, str):
        return value
    return "[" + ",".join(str(item) for item in value) + "]"


overrides = [
    "agent=WoTE_agent",
    "agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig",
    f"++agent.config.cluster_file_path={q(data.get('planner_anchors'))}",
    f"++agent.config.controller_style_ref_bank_path={q(controller.get('style_ref_path', controller.get('ref_path')))}",
    f"++agent.config.controller_style_exec_bank_path={q(controller.get('style_exec_path', controller.get('exec_path')))}",
    f"++agent.config.controller_candidate_exec_bank_path={q(controller.get('candidate_exec_path', controller.get('exec_path')))}",
    f"use_cache_without_dataset={bool_s(data.get('use_cache_without_dataset', True))}",
    f"cache_path={q(data.get('cache_path'))}",
    f"dataloader.params.batch_size={optim.get('batch_size', 16)}",
    f"dataloader.params.num_workers={optim.get('num_workers', 2)}",
    f"trainer.params.max_epochs={optim.get('max_epochs', 32)}",
    f"+trainer.params.devices={devices}",
    f"trainer.params.precision={q(optim.get('precision', '16-mixed'))}",
    f"trainer.params.strategy={q(optim.get('strategy', 'ddp'))}",
    f"split={q(data.get('split', 'trainval'))}",
    f"experiment_name={q('WoTE/' + run_name)}",
    f"scene_filter={q(data.get('scene_filter', 'navtrain'))}",
    f"agent.lr={optim.get('lr', '1e-4')}",
    f"agent.config.min_lr={optim.get('min_lr', '1e-6')}",
    f"agent.config.warmup_epochs={optim.get('warmup_epochs', 3)}",
    f"++agent.config.freeze_all={bool_s(freeze.get('freeze_all', True))}",
    f"++agent.config.trainable_groups={q(list_s(freeze.get('trainable_groups', [])))}",
    f"++agent.config.trainable_prefixes={q(list_s(freeze.get('trainable_prefixes', [])))}",
    f"++agent.config.frozen_prefixes={q(list_s(freeze.get('frozen_prefixes', [])))}",
    f"++agent.config.freeze_strict={bool_s(freeze.get('strict', True))}",
    f"++agent.config.print_trainable={bool_s(freeze.get('print_trainable', False))}",
    f"++agent.config.use_agent_loss={bool_s(losses.get('agent', False))}",
    "++agent.config.use_map_loss=true",
    f"++agent.config.bev_semantic_weight={losses.get('current_bev', 0.0)}",
    f"++agent.config.fut_bev_semantic_weight={losses.get('future_bev', 0.0)}",
    f"++agent.config.traj_offset_loss_weight={losses.get('traj_offset', 0.0)}",
    f"++agent.config.offset_im_reward_weight={losses.get('offset_im_reward', 0.0)}",
    f"++agent.config.im_loss_weight={losses.get('imitation_reward', 0.0)}",
    f"++agent.config.metric_loss_weight={losses.get('metric_reward', 0.0)}",
    f"++agent.config.herm_enable={bool_s(switches.get('herm_enable', False))}",
    f"++agent.config.herm_apply_in_train={bool_s(switches.get('herm_apply_in_train', False))}",
    f"++agent.config.herm_apply_in_eval={bool_s(switches.get('herm_apply_in_eval', False))}",
    f"++agent.config.use_offset_candidates_in_train={bool_s(switches.get('use_offset_candidates_in_train', False))}",
    f"++agent.config.use_scored_candidates_for_fut_bev_target={bool_s(switches.get('use_scored_candidates_for_fut_bev_target', False))}",
    f"++agent.config.use_scored_candidates_for_im_loss={bool_s(switches.get('use_scored_candidates_for_im_loss', False))}",
    f"++agent.config.use_controller_wm={bool_s(controller.get('use_world_model', True))}",
    f"++agent.config.controller_wm_fusion={q(controller.get('fusion', 'attn_film'))}",
    f"++agent.config.controller_wm_token_scope={q(controller.get('token_scope', 'all'))}",
    f"++agent.config.controller_wm_first_step_only={bool_s(controller.get('first_step_only', False))}",
    f"++agent.config.controller_feature_mode={q(controller.get('feature_mode', 'full'))}",
]

print("HYDRA_OVERRIDES=(")
for override in overrides:
    if override.endswith("=None") or override.endswith("=''"):
        continue
    print(f"  {override}")
print(")")
PY
)

eval "${_generated}"

export PYTHONPATH="${ROOT}:${ROOT}/navsim:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"

echo "[INFO] Config: ${WOTE_YAML_CONFIG}"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] PYTHON=${PYTHON}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[INFO] NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-}"
echo "[INFO] Overrides:"
printf '  %s\n' "${HYDRA_OVERRIDES[@]}" "$@"

if [[ "${WOTE_DRY_RUN:-0}" == "1" ]]; then
  echo "[INFO] WOTE_DRY_RUN=1, not launching training."
  exit 0
fi

cd "${ROOT}"
exec "${PYTHON}" ./navsim/planning/script/run_training.py "${HYDRA_OVERRIDES[@]}" "$@"

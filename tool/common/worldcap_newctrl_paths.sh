#!/usr/bin/env bash

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}

export WORLDCAP_DATA_ROOT=${WORLDCAP_DATA_ROOT:-"${ROOT}/newCtrl"}

export WORLDCAP_CTRL_REF_64=${WORLDCAP_CTRL_REF_64:-"${WORLDCAP_DATA_ROOT}/controller/ref_trajs/Anchors_Original_64_centered.npy"}
export WORLDCAP_CTRL_EXEC_64=${WORLDCAP_CTRL_EXEC_64:-"${WORLDCAP_DATA_ROOT}/controller/bundles/64/controller_styles_64.npz"}

export WORLDCAP_CTRL_REF_128=${WORLDCAP_CTRL_REF_128:-"${WORLDCAP_DATA_ROOT}/controller/ref_trajs/Anchors_Original_128_centered.npy"}
export WORLDCAP_CTRL_EXEC_128=${WORLDCAP_CTRL_EXEC_128:-"${WORLDCAP_DATA_ROOT}/controller/bundles/128/controller_styles_128.npz"}

export WORLDCAP_CTRL_REF_1024=${WORLDCAP_CTRL_REF_1024:-"${WORLDCAP_DATA_ROOT}/controller/ref_trajs/Anchors_Original_1024_centered.npy"}
export WORLDCAP_CTRL_EXEC_1024=${WORLDCAP_CTRL_EXEC_1024:-"${WORLDCAP_DATA_ROOT}/controller/bundles/1024/controller_styles_1024.npz"}

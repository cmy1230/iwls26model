#!/usr/bin/env bash
# 在下面数组里写上要训练的电路名（子串匹配，与 train.py --circuit 一致）
# 运行: chmod +x train_circuits.sh && ./train_circuits.sh
# 日志: ./log/<电路名>.log

set -uo pipefail

# ---------- 你改这里 ----------
PYTHON="${PYTHON:-python3}"
TRAIN="${TRAIN:-train.py}"          # 与脚本同目录下的 train.py
LOG_DIR="${LOG_DIR:-./log}"
TRAIN_EXTRA=""                      # 需要时写上，例如: TRAIN_EXTRA="--max_epochs 500"

CIRCUITS=(
  "or1200"
  # "aes"
  # "ibex"
)
# ----------------------------

if [[ ${#CIRCUITS[@]} -eq 0 ]]; then
  echo "错误: CIRCUITS 数组为空，请编辑本脚本填写电路名。" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
mkdir -p "$LOG_DIR"

failed=0
for c in "${CIRCUITS[@]}"; do
  [[ -z "${c}" ]] && continue

  safe="${c//\//_}"
  log="${LOG_DIR}/${safe}.log"
  echo "======== $(date -u +"%Y-%m-%dT%H:%M:%SZ")  ${c} -> ${log} ========" | tee -a "${log}"

  # shellcheck disable=SC2086
  "${PYTHON}" "${TRAIN}" --circuit "${c}" ${TRAIN_EXTRA} >>"${log}" 2>&1
  ec=$?

  if [[ "${ec}" -eq 0 ]]; then
    echo "[OK] ${c}" | tee -a "${log}"
  else
    echo "[FAIL] ${c} exit=${ec}" | tee -a "${log}"
    failed=$((failed + 1))
  fi
  echo "" | tee -a "${log}"
done

[[ "${failed}" -gt 0 ]] && exit 1
exit 0

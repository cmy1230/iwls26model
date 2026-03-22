LOG_DIR="${LOG_DIR:-./log}"

CIRCUITS=(
  "ac97_ctrl"
  "aes"
  "aes_secworks"
  "aes_xcrypt"
  "des3_area"
  "dft"
  "dynamic_node"
  "ethernet"
  "fir"
  "fpu"
  "idft"
  "iir"
  "jpeg"
  "pci"
  "sasc"
  "sha256"
  "simple_spi"
  "spi"
  "ss_pcm"
  "tv80"
  "usb_phy"
  "vga_lcd"
  "wb_conmax"
  "wb_dma"
  "bgm"
  "blob_merge"
  "boundtop"
  "LU32PEEng"
  "LU64PEEng"
  "LU8PEEng"
  "mcml"
  "mkDelayWorker32B"
  "mkPktMerge"
  "mkSMAdapter4B"
  "or1200"
  "raygentop"
  "sha"
  "spree"
  "stereovision0"
  "stereovision1"
  "stereovision2"
  "stereovision3"
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
  python train.py --circuit "${c}"  >>"${log}" 2>&1
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

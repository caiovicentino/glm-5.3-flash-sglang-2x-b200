#!/usr/bin/env bash
# GLM-5.3-Flash-NVFP4 on 2x NVIDIA B200 (SM100) with SGLang — the flags we validated on 2026-09-11.
# Usage: API_KEY=... MODEL_PATH=/root/model-glm53 bash serve.sh
# Env knobs: TP (2) CONTEXT_LENGTH (262144) CHUNK (8192) CONC (24) MEMF (0.85) ATTN (dsa) MOE (auto)
#            SERVED_MODEL_NAME (glm-5.3-flash) ADAPTIVE_SPEC (./adaptive_spec.json) EXTRA_ARGS PORT (8000)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/model-glm53}"
KEY="${API_KEY:-$(cat /root/.api_key 2>/dev/null || true)}"
[ -n "$KEY" ] || { echo "API_KEY not set (export API_KEY=... or write /root/.api_key)"; exit 1; }

# The NVFP4 requant ships a chat template that silently disables vision; use the original one.
if ! grep -q "emit_image" "$MODEL_PATH/chat_template.jinja" 2>/dev/null; then
  cp -n "$MODEL_PATH/chat_template.jinja" "$MODEL_PATH/chat_template.jinja.requant" 2>/dev/null || true
  cp -f "$HERE/chat_template.jinja" "$MODEL_PATH/chat_template.jinja"
  echo "chat_template.jinja replaced with the vision-enabled template (backup: chat_template.jinja.requant)"
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash}" \
  --tp-size "${TP:-2}" --ep-size 1 \
  --context-length "${CONTEXT_LENGTH:-262144}" \
  --quantization modelopt_fp4 \
  --attention-backend "${ATTN:-dsa}" \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  ${MOE:+--moe-runner-backend "$MOE"} \
  --disable-shared-experts-fusion \
  --chunked-prefill-size "${CHUNK:-8192}" --max-prefill-tokens "${CHUNK:-8192}" \
  --max-running-requests "${CONC:-24}" --mem-fraction-static "${MEMF:-0.85}" \
  --cuda-graph-max-bs-decode "${CONC:-24}" --cuda-graph-bs-decode 1 2 4 8 12 16 24 \
  --speculative-algorithm NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --speculative-adaptive --speculative-adaptive-config "${ADAPTIVE_SPEC:-$HERE/adaptive_spec.json}" \
  --mamba-track-interval 64 \
  --mm-attention-backend triton_attn \
  --enable-multimodal --enable-metrics --media-url-max-file-size-mb 1024 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  ${EXTRA_ARGS:-} \
  --api-key "$KEY" --host 0.0.0.0 --port "${PORT:-8000}"

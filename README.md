# GLM-5.3-Flash-NVFP4 on 2× NVIDIA B200 with SGLang

The recipe we validated on 2026-09-11 for serving **GLM-5.3-Flash-NVFP4** (320B total / 18B active,
vision, DSA sparse attention, NextN speculative decoding) on a **pair of B200** (SM100, 183 GB each,
NVLink) with SGLang, and the numbers it produced next to our production box, a 4× RTX PRO 6000
(SM120, PCIe) running the same checkpoint with the sibling recipe
[glm-5.3-flash-sglang-4x-rtx-pro-6000](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-rtx-pro-6000).

Short version: same model, same image, same client-facing behaviour, and the B200 pair is **2× faster
for one user, 2.3× at 8 concurrent requests, 4.5× at 24, with 3× the prefill rate**, at about twice
the hourly price. The gap widens with concurrency because the SM120 box is bound by per-step latency
(small kernels + PCIe collectives), which NVLink and the native SM100 kernels remove.

## Recipe card

| | |
|---|---|
| hardware | 2× B200 183 GB (NVLink NV18), Xeon 128 vCPU, 774 GB RAM, 500 GB NVMe — a Vast.ai "verified datacenter" offer at US$ 15.5/h |
| image | `lmsysorg/sglang:glm-5.3-flash` (SGLang `0.0.0.dev1+gf13cb6f6a7`, torch 2.13+cu130). Launch the instance from this image, or extract it without Docker with `tools/pull_img.py` |
| checkpoint | `LibertAIDAI/GLM-5.3-Flash-NVFP4` rev `9e0d74e3` (182 GiB): routed experts NVFP4 g16, attention/embeddings/lm_head BF16, vision tower BF16 |
| patches | the four **loader** files from 0xSero (`patches/`), without them `--tp-size 2` fails in `load_qkv_weight`; his two SM120 kernel shims are not needed here |
| chat template | `serve/chat_template.jinja` — the original `zai-org/GLM-5.3-Flash` template. The requant's own template silently disables vision (it injects a "you are unable to process this image" reminder). `serve.sh` swaps it in |
| parallelism | `--tp-size 2 --ep-size 1` |
| attention | `--attention-backend dsa`; SM100 auto-selects `trtllm` for DSA prefill and decode with `--kv-cache-dtype fp8_e4m3`; `--linear-attn-backend triton` for the KDA layers; `--mm-attention-backend triton_attn` for the vision tower |
| MoE | auto = `flashinfer_trtllm` (the image picks it on SM100 and sets `--disable-shared-experts-fusion`) — no Marlin, no SM120 workarounds |
| speculation | NextN, `--speculative-num-steps 5 --speculative-num-draft-tokens 6`, adaptive with one candidate per batch bucket (`serve/adaptive_spec.json`: 5 steps at batch 1, 3 steps at batch ≥ 2) |
| memory | `--mem-fraction-static 0.85` → KV pool **4.9 M tokens** (fp8) vs 1.85 M on the 4× SM120 box |
| prefill | `--chunked-prefill-size 8192` |
| concurrency | `--max-running-requests 24`, decode CUDA graphs for batch 1 2 4 8 12 16 24 |
| context | `--context-length 262144` for the test (the checkpoint supports 1 M; the KV pool has room) |
| client | OpenAI-compatible on :8000, `--reasoning-parser glm45 --tool-call-parser glm47`, `--enable-multimodal` |

## Quick start

```bash
# 1. machine launched from lmsysorg/sglang:glm-5.3-flash (or: python3 tools/pull_img.py, see tools/README.md)
git clone https://github.com/caiovicentino/glm-5.3-flash-sglang-2x-b200 && cd glm-5.3-flash-sglang-2x-b200
REF=<0xSero commit> bash patches/fetch_patches.sh          # 4 files; then:
python -c "import sglang.srt.models.glm5_next"               # must succeed
hf download LibertAIDAI/GLM-5.3-Flash-NVFP4 --revision 9e0d74e3 --local-dir /root/model-glm53   # 182 GiB
API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))') MODEL_PATH=/root/model-glm53 bash serve/serve.sh
```

Boot is ~4 min (weights ~1 min from NVMe, then CUDA-graph capture for 7 decode batch sizes and the
verify graphs). Warm the server before measuring: the first request of each new prompt shape pays
~18 s of DeepGEMM/FlashInfer JIT compilation, after which TTFT is sub-second (see below).

## Measured (2026-09-11, idle server, same scripts as the SM120 recipe)

| axis | 4× RTX PRO 6000 (production) | 2× B200 |
|---|---|---|
| pt-BR prose, 1 request (`bench/prose_speed.py`, 8 × 600 tokens) | ~100 tok/s | **215 tok/s** |
| code, 1 request (`bench/code_speed.py`) | 167–197 tok/s | **261–284 tok/s** |
| 8 concurrent × 400 tokens (`bench/concurrency.py 8 400`) | 472–508 tok/s aggregate | **1,117 tok/s** |
| 16 concurrent | — | **1,773 tok/s** |
| 24 concurrent | 542 tok/s (batch 12, its ceiling) | **2,420 tok/s** (101 tok/s per stream) |
| warm TTFT, 4k / 14k / 26k prompt tokens (`bench/ttft.py`) | ~0.5 / 1.7 / 3.2 s | **0.24 / 0.6 / 1.1 s** |
| prefill rate (cold prompt, warm kernels) | 8.1–8.4k tok/s | **22–26k tok/s** |
| NextN acceptance, prose | 3.5–4 draft tokens | 2.5–2.9 |
| KV pool | 1.85 M tokens | 4.9 M tokens |

Functional gates (`bench/gates.py`): API, tool calling, thinking on/off (the "opencode form"), vision
(solid colour, text in image, shapes) and 8-stream concurrency pass. Two gates report false negatives on
this build: `usage.prompt_tokens_details.cached_tokens` is not populated even though the prefix cache
works (server metrics: 8,064 of 17,675 prompt tokens served from cache), and the strict string match of
the shapes gate rejects a correct answer. One gate is a budget effect, not a defect: with the template's
default `reasoning_effort` (max) the model spends 6,000–8,000 tokens reasoning about an essay prompt
before writing — coherent, no repetition, just long. Send `reasoning_effort: high` (or `low`) from the
client; see the sibling recipe's `docs/reasoning-effort.md`.

## Pitfalls we hit (so you don't)

- **`--tp-size 2` fails to load without the loader patches** (`AssertionError: param_data.shape=[4096, 2048]`
  vs `loaded_weight.shape=[4096, 4096]` in `load_qkv_weight`). Apply `patches/` before anything else.
- **Vision silently off**: the requant's `chat_template.jinja` replaces every image with a text reminder;
  the request "works" and the model says it cannot see images (`Warning: More image data items provided
  than corresponding tokens found in the prompt` in the log). `serve.sh` installs the original template.
- **Adaptive spec format**: `{"1": {"candidate_steps": [5], "up_hysteresis": 0.0, "down_hysteresis": 0.0,
  "ceiling_coeff": 0}, ...}`. The short form `{"1": [5]}` crashes the scheduler on this build.
- **JIT warm-up**: first request per prompt shape ≈ 18 s TTFT; not a regression, warm the server.
- **Docker Hub throttling**: extracting the image at 1–3 MB/s while `hf download` runs; do one at a time.
- **Disk**: 182 GiB checkpoint + 14 GB image; a 500 GB NVMe leaves no room for a second large model.

## What we did not do

- No SM120 flags: no `marlin`, no `flashinfer_sparse_mla`, no `--disable-custom-all-reduce`.
- No PD disaggregation, no DP attention, no EP > 1 — with two GPUs on NVLink, plain TP2 was enough.
- No 1 M-context test; the KV pool would take it, we stopped at 262k.
- No long soak: this was a 3-hour measurement session, not days of production.

## Layout

```
serve/serve.sh              launcher (all knobs are env vars)
serve/adaptive_spec.json    one NextN step count per batch bucket
serve/chat_template.jinja   vision-enabled template (original zai-org/GLM-5.3-Flash)
serve/env.example
patches/                    fetch_patches.sh (4 loader files from 0xSero) + README
tools/                      pull_img.py + img_manifest.json (extract the image without Docker)
bench/                      gates.py, prose_speed.py, code_speed.py, concurrency.py, ttft.py, per_batch.sh
docs/results.md             raw numbers and how they were taken
docs/incidents.md           what broke, in order
```

## Credits

0xSero for the loader patches and the first working GLM-5.3-Flash recipe on consumer Blackwell;
LibertAI for the NVFP4 checkpoint; zai-org for GLM-5.3-Flash and its chat template; the SGLang team.
MIT license; the checkpoint and template keep their own licenses.

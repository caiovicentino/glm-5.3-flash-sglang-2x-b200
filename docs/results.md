# Results — 2× B200, 2026-09-11 (00:30–02:30 UTC−3)

Server idle between runs; each script run at least twice; numbers are the second (warm) run.
Production column: 4× RTX PRO 6000, same checkpoint, sibling recipe, measured 2026-09-04…09-10.

## Single request

| script | 2× B200 | 4× RTX PRO 6000 |
|---|---|---|
| `prose_speed.py` (8 × 600 tokens pt-BR, thinking off) | 4,800 tokens in 22–23 s → **215 tok/s** | ~45 s → ~105 tok/s |
| `code_speed.py` (4 × 700 tokens) | 2,800 tokens in 10–11 s → **261–284 tok/s** | 167–197 tok/s (from server logs) |
| NextN acceptance (server log, prose) | 2.5–2.9 | 3.5–4.0 |

## Concurrency (`concurrency.py N 400`, thinking off, distinct prompts)

| N | 2× B200 aggregate | per stream | 4× RTX PRO 6000 |
|---|---|---|---|
| 8 | 1,117 tok/s | 140 | 472–508 |
| 16 | 1,773 tok/s | 111 | — |
| 24 | 2,420 tok/s | 101 | 542 (batch 12) |

`gates.py` gate F (8 streams, 400 tokens each): 704 / 867 / 897 tok/s across three runs.

## Prefill / TTFT (`ttft.py`, cold prompt = unique salt, streaming, first content token)

| prompt tokens | run 1 (cold kernels) | run 2 | run 3 | ≈ prefill tok/s |
|---|---|---|---|---|
| 1,070 | 18.2 s | 0.24 s | 0.79 s | 1.3–4.4k (dominated by fixed cost) |
| 4,008 | 18.6 s | 0.24 s | 0.26 s | 15–17k |
| 7,209 | 0.32 s | 0.29 s | 0.28 s | 25k |
| 14,183 | 1.48 s | 0.61 s | 0.54 s | 23–26k |
| 26,666 | 19.6 s | 1.19 s | 1.10 s | 22–24k |

Run 1's 18–20 s entries are DeepGEMM/FlashInfer JIT compilation for a prompt shape not seen before.
Production (4× SM120): 8.1–8.4k tok/s in full 8k chunks.

## Memory

`KV Cache is allocated. dtype: torch.float8_e4m3fn`, `max_total_num_tokens=4,900,608` at
`--mem-fraction-static 0.85` and 262k context. Production: 1.85 M tokens at 0.80.

## Gates (`gates.py`, third run, with the vision template)

| gate | result | note |
|---|---|---|
| A models | pass | |
| B long generation | fail | 8,000-token budget consumed by reasoning at default effort (max); coherent, no repetition; passes with `reasoning_effort: high` |
| D opencode form | pass | content + reasoning split correctly |
| E tool call | pass | |
| H1 colour | pass | "Vermelho" |
| H2 text in image | pass | "CB4271" |
| H3 shapes | fail | answer is correct ("Quadrado azul … superior esquerda …"); the gate's string match is too strict |
| G prefix cache | fail | `usage.cached_tokens` = 0 while server metrics show 8,064/17,675 cached; reporting only |
| F 8 streams | pass | 867 tok/s |

## Reasoning length at default effort

Prompt: "Escreva um texto longo e detalhado, em português do Brasil, sobre a história de Minas Gerais no século XVIII."

| `chat_template_kwargs` | reasoning | content |
|---|---|---|
| `{"enable_thinking": true}` (= max) | 6,000 tokens and still going (21k chars, zero repeated 10-grams, a numbered outline past item 67) | empty at the 6,000 cap |
| `{"enable_thinking": true, "reasoning_effort": "high"}` | 484 chars | 8,583 chars, correct pt-BR with accents |
| `{"enable_thinking": true, "reasoning_effort": "low"}` | 0 | 10,453 chars |
| `{"enable_thinking": false}` | 0 | the English plan leaks into the content (known; worst option) |

# Incidents, in order (2026-09-11)

1. **Image extraction crawled at 1–3 MB/s** while `hf download` used the 800 Mbps link; Docker Hub's
   CDN lost the contention. It finished at 17 MB/s once the checkpoint download ended. Sequence them.

2. **Scheduler crash at boot:** `AttributeError: 'list' object has no attribute 'get'` in
   `adaptive_spec_params._load_adaptive_config`. Cause: adaptive spec written in the short form
   `{"1": [5], "2": [3]}`. This build wants `{"1": {"candidate_steps": [5], "up_hysteresis": 0.0,
   "down_hysteresis": 0.0, "ceiling_coeff": 0}, ...}` (`serve/adaptive_spec.json`).

3. **Weight loading fails at TP2** on the stock image:
   `AssertionError: param_data.shape=torch.Size([4096, 2048]), loaded_weight.shape=torch.Size([4096, 4096])`
   from `glm5_next.load_weights → weight_loader_v2 → load_qkv_weight`. Fixed by the four loader
   patches from 0xSero (`patches/`). The two SM120 kernel files of his set were reverted to stock and
   are not needed on SM100.

4. **Vision "works" but the model says it cannot see images.** Log: `Warning: More image data items
   provided than corresponding tokens found in the prompt`; the image request tokenizes to 51 prompt
   tokens (no image tokens). Not the attention backend (`fa4` vs `triton_attn` made no difference):
   the requant's `chat_template.jinja` (8,617 bytes, rev 9e0d74e3) renders every media item as a
   `<reminder>` and never emits `<|begin_of_image|><|image|><|end_of_image|>`. The original
   `zai-org/GLM-5.3-Flash` template (10,644 bytes, macro `emit_image`) fixes it; `serve.sh` installs it.

5. **First TTFT measurements showed 18–20 s** for 1k, 4k and 26k prompts and 0.3 s for 7k: JIT
   compilation of DeepGEMM/FlashInfer kernels per new shape. Re-running gave 0.24–1.2 s. Warm before
   measuring (or before putting users on it).

6. **Gate B (long pt-BR generation) fails three times** with 8,000 tokens and zero accents: the whole
   budget went to reasoning at the template's default effort (max). The reasoning is coherent (no
   repeated 10-gram in 21k chars). Not a numerics problem; a client-side `reasoning_effort` matter.

Operational note: `pkill -f "[s]glang"` will kill your own SSH session if the remote command line
contains the literal `sglang` anywhere (e.g. `/opt/sglang/bin/python`). Kill by `[l]aunch_server`
instead, and keep launches and checks in separate SSH calls.

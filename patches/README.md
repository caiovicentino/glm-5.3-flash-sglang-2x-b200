# Loader patches (four files from 0xSero)

The stock image `lmsysorg/sglang:glm-5.3-flash` does not load `LibertAIDAI/GLM-5.3-Flash-NVFP4`
at `--tp-size 2`. It dies in weight loading:

```
File "srt/models/glm5_next.py", line 1691, in load_weights
File "srt/layers/linear.py", line 1183, in weight_loader_v2  ->  param.load_qkv_weight(
AssertionError: param_data.shape=torch.Size([4096, 2048]), loaded_weight.shape=torch.Size([4096, 4096])
```

The four loader-side files from https://github.com/0xSero/glm-5.3-flash-sglang-sm120 fix it (they are
his work; `fetch_patches.sh` downloads them at the commit you pin with `REF=`):

| file in the SGLang tree | what it does |
|---|---|
| `srt/layers/quantization/modelopt_quant.py` | slicing of NVFP4 expert scales |
| `srt/layers/quantization/utils.py` | quantization utilities |
| `srt/models/glm5_next.py` | model file fixes (the QKV loader path above) |
| `srt/models/deepseek_nextn.py` | the NextN (MTP) layer for this checkpoint |

His two other files (`flash_mla_sm120.py`, `dsa_backend.py`) are SM120 kernel shims and are **not**
applied on B200: SM100 uses the native `trtllm` DSA prefill/decode kernels that the image already has.

Apply, then verify before downloading anything else:

```
REF=<commit> bash patches/fetch_patches.sh
python -c "import sglang.srt.models.glm5_next"
```

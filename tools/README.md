# tools

`pull_img.py` + `img_manifest.json`: extract `lmsysorg/sglang:glm-5.3-flash` (amd64) **without Docker**
into `/root/sgl-img`, verifying each layer's sha256 and honoring OCI whiteouts. Useful on hosts where
you get a container but no Docker daemon (Vast.ai instances launched from another image). After
extraction, point `/opt/sglang` and `/sgl-workspace` at the extracted tree:

```
python3 tools/pull_img.py                      # ~14 GB, 75 layers
mv /opt/sglang /opt/sglang.orig; ln -s /root/sgl-img/opt/sglang /opt/sglang
mv /sgl-workspace /sgl-workspace.orig; ln -s /root/sgl-img/sgl-workspace /sgl-workspace
/opt/sglang/bin/python -c "import sglang; print(sglang.__version__)"   # 0.0.0.dev1+gf13cb6f6a7
```

If you can launch the instance directly from the `lmsysorg/sglang:glm-5.3-flash` image, you do not
need this: the tree is already at `/sgl-workspace/sglang` and the venv at `/opt/sglang`.

Docker Hub throttles anonymous layer downloads to a few MB/s while another download saturates the
link; run the extraction before (or after) `hf download`, not during.

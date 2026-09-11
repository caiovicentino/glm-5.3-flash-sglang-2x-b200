#!/usr/bin/env python3
"""Puxa e extrai lmsysorg/sglang:glm-5.3-flash (amd64) sem Docker.
Verifica sha256 de cada camada. Honra whiteouts OCI (.wh.*)."""
import hashlib, json, os, shutil, sys, tarfile, time, urllib.request, urllib.error

REPO="lmsysorg/sglang"; DEST="/root/sgl-img"; BLOBS="/root/sgl-blobs"
man=json.load(open("/root/img_manifest.json"))

def token():
    with urllib.request.urlopen(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull",
        timeout=60) as h: return json.loads(h.read())["token"]

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**k): return None

def fetch_blob(dig, path):
    """Baixa com auth; segue redirect para CDN SEM o header (senao 400)."""
    url=f"https://registry-1.docker.io/v2/{REPO}/blobs/{dig}"
    op=urllib.request.build_opener(NoRedirect)
    req=urllib.request.Request(url); req.add_header("Authorization",f"Bearer {token()}")
    try:
        resp=op.open(req,timeout=120)
    except urllib.error.HTTPError as e:
        if e.code in (301,302,303,307,308):
            resp=urllib.request.urlopen(e.headers["Location"],timeout=300)
        else: raise
    h=hashlib.sha256()
    with open(path,"wb") as f:
        while True:
            c=resp.read(4*1024*1024)
            if not c: break
            f.write(c); h.update(c)
    got="sha256:"+h.hexdigest()
    if got!=dig: raise SystemExit(f"❌ DIGEST NAO BATE\n   esperado {dig}\n   obtido   {got}")
    return os.path.getsize(path)

def apply_whiteouts(tf, dest):
    for m in tf.getmembers():
        b=os.path.basename(m.name)
        if b.startswith(".wh."):
            if b==".wh..wh..opq":
                d=os.path.join(dest,os.path.dirname(m.name))
                if os.path.isdir(d):
                    for e in os.listdir(d):
                        p=os.path.join(d,e)
                        shutil.rmtree(p,ignore_errors=True) if os.path.isdir(p) else os.remove(p)
            else:
                p=os.path.join(dest,os.path.dirname(m.name),b[4:])
                if os.path.isdir(p): shutil.rmtree(p,ignore_errors=True)
                elif os.path.exists(p): os.remove(p)

os.makedirs(DEST,exist_ok=True); os.makedirs(BLOBS,exist_ok=True)
t0=time.time(); baixado=0
layers=man["layers"]
for i,l in enumerate(layers):
    dig=l["d"]; short=dig[7:19]; bp=os.path.join(BLOBS,short)
    if l["s"]==0 or l["s"]<512:
        print(f"[{i+1:2d}/{len(layers)}] {short} vazia, pulando",flush=True); continue
    if not os.path.exists(bp) or os.path.getsize(bp)!=l["s"]:
        fetch_blob(dig,bp)
    baixado+=l["s"]
    try:
        with tarfile.open(bp,"r:*") as tf:
            apply_whiteouts(tf,DEST)
        with tarfile.open(bp,"r:*") as tf:
            safe=[m for m in tf.getmembers() if not os.path.basename(m.name).startswith(".wh.")]
            tf.extractall(DEST,members=safe,filter="tar")
    except Exception as e:
        print(f"   ⚠️ camada {short}: {type(e).__name__}: {e}",flush=True)
    el=time.time()-t0
    print(f"[{i+1:2d}/{len(layers)}] {short} {l['s']/2**20:8.1f} MiB · "
          f"{baixado/2**30:5.2f} GiB · {baixado/2**20/max(el,1):.0f} MiB/s",flush=True)

print(f"\n✅ concluido em {(time.time()-t0)/60:.1f} min",flush=True)
os.system(f"du -sh {DEST}; df -h / | tail -1")

#!/usr/bin/env python3
"""Strata prefetch for NixOS: download, verify and prepare the model files, then emit a run config.

    strata-prefetch --family swift --model IQ2_XS --context 32768 --data-dir /var/lib/strata \
        --emit-config /var/lib/strata/config.json --engine /run/current-system/.../bin/strata

Families: the Flash-Next GGUF packs (qwen, swift, abliterated, coder, unsloth, gyro), the
ready-made NVFP4 packs, the own-path q36, Tencent Hy4-Flash, and Inkling-2-Mini.  For
everything upstream knows, this tool imports the packaged setup.py and strata_control and uses
its own tables, fetchers, verifiers, fit rules and tokenizer routes: the pins, shard layouts,
file tables and engine arguments are upstream's, not transcriptions.  The abliterated family
(SC117's re-quant of Swift 1.5, not in upstream's FAMILIES) keeps its own table.

Steps (each skipped when already done):
  1. download the family's files (resumable, size- and SHA-256-verified, at pinned revisions)
  2. build the pack with upstream tools (iq_pack.py), when the family needs one
  3. fetch the MTP draft layer / the DSpark drafter, when the family uses one
  4. with images on: the picture reader (mmproj for the GGUF families, the Vision .strata for
     Hy4-Flash and Inkling-2-Mini, which the engine reads itself)
  5. build the tokenizer (strata_tokenizer.py, or Section 0 of the .strata file)
  6. write prepared.json and, with --emit-config, the engine run config for serve/server.py

No model weights are installed by nix: they live in --data-dir.  A fully prepared
data dir needs no network (the .done marks short-circuit every download).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SHARE = Path(os.environ.get("STRATA_SHARE", "")) if os.environ.get("STRATA_SHARE") else None
if SHARE is None:
    sys.exit("strata-prefetch: STRATA_SHARE is not set (run it through the strata package's wrapper)")

TOOLS = SHARE / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(SHARE))          # strata_control lives beside setup.py in the share

# The packaged setup.py is the source of truth: its FAMILIES/MODELS/revision pins, its
# download/verify functions, its family fetchers, its tokenizer builders, and the fit modules.
_spec = importlib.util.spec_from_file_location("strata_setup", str(SHARE / "setup.py"))
S = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(S)
except Exception as e:                  # a broken share must say so plainly
    sys.exit(f"strata-prefetch: the packaged setup.py cannot be loaded ({e})")

from strata_control import hy4_family, inkling_family, hy4_fit, inkling_fit

HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
CONTEXTS_ALL = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288, 786432, 1048576]
ISTA = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
SC117 = "SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
ABLITERATED_REV = "3db919a964b1f0ae0d1000ba7be90b3e38b1f7a3"

# The GGUF-pack families at the revisions setup.py pins; qwen/swift/coder read their revision
# straight from setup.py, so a flake bump updates them. abliterated (SC117's re-quant, not in
# upstream's FAMILIES) keeps its own pins: its picture reader is SC117's own file (same name as
# the original's, 64 bytes shorter, so its sha is ours, not the original's alias). The coder
# family shares the original's picture reader (upstream: its shard 2 and vision encoder are the
# original's files).
def _pack_family(models, **kw):
    return dict(models=models, **kw)

# The models a family ships: the MODELS entries whose families field names it (upstream's own
# rule, the same one unsloth/gyro/nvfp4 use below).
def models_of(family: str) -> list:
    return [m for m, d in S.MODELS.items() if d.get("families") == (family,)]

FAMILIES = {
    "qwen": {
        "repo": ISTA,
        "revision": S.HF_REVISIONS[ISTA],
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "shards": 2,
        "tag": "",
        "name": "qwen3.8-flash-next",
        "title": "Qwen3.8-Flash-Next",
        "profile": "expert-profile.bin",
        "models": ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"],
        "vision_file": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "vision_repo": ISTA,
        "vision_hf": f"https://huggingface.co/{ISTA}/resolve/{S.HF_REVISIONS[ISTA]}/",
    },
    "swift": {
        "repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "revision": S.HF_REVISIONS["ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"],
        "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "shards": 2,
        "tag": "swift-",
        "name": "swift-1.5",
        "title": "Swift 1.5",
        "profile": "expert-profile.bin",
        # the quants ukisai publishes at the repo root (its tree: Q2_0, IQ2_XS, IQ3_XXS, IQ3_S,
        # IQ3_S_G8; upstream's own catalog names Q2_0/IQ2_XS/IQ3_XXS for swift)
        "models": ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"],
        "vision_file": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",
        "vision_repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "vision_hf": ("https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/"
                      "resolve/" + S.HF_REVISIONS["ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"] + "/"),
    },
    "abliterated": {
        "repo": SC117,
        "revision": ABLITERATED_REV,
        "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-{q}-0000{i}-of-00002.gguf",
        "shards": 2,
        "tag": "abliterated-",
        "name": "swift-1.5-abliterated",
        "title": "Swift 1.5 abliterated",
        "profile": "expert-profile.bin",
        # SC117's tree ships Q2_0, IQ2_XS, IQ3_XXS in per-quant folders, IQ3_S at the root, and
        # its own picture reader; the pack route is the same as swift's, at SC117's own pin
        "models": ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"],
        # the quants SC117 publishes at the repo root, the rest in their own folder
        "root_models": ["IQ3_S"],
        "vision_file": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "vision_hf": f"https://huggingface.co/{SC117}/resolve/{ABLITERATED_REV}/",
        # SC117 publishes its own picture reader beside the text shards: the same name as the
        # original's, 64 bytes shorter, so its sha is ours. Its exact size is not published in
        # upstream's tables (the repo is not in setup.py), so this file is verified by sha only.
        "vision_pin": (0, "2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355"),
    },
    "coder": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
        "revision": S.HF_REVISIONS["ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"],
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "shards": 2,
        "tag": "coder-",
        "name": "qwen3.8-flash-next-coder",
        "title": "Qwen3.8-Flash-Next Coder",
        "profile": "expert-profile-coder.bin",
        # the coder repo ships IQ1_M only (its tree has one quant folder); upstream's MODELS
        # table names the same set, so the family's models come from it
        "models": models_of("coder"),
        # the coder's picture reader is the original's file, at the original's pin
        "vision_file": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "vision_repo": ISTA,
        "vision_hf": f"https://huggingface.co/{ISTA}/resolve/{S.HF_REVISIONS[ISTA]}/",
    },
}

# Unsloth, Gyro and NVFP4: upstream's own family tables (hf pattern, shard tables, sha pins,
# pack_args --compat-bf16, the ISTA picture reader in mmproj_hf). The models a family ships are
# the MODELS entries whose families field names it.
def models_of(family: str) -> list:
    return [m for m, d in S.MODELS.items() if d.get("families") == (family,)]

FAMILIES["unsloth"] = dict(S.FAMILIES["unsloth"], profile="expert-profile.bin",
                           models=models_of("unsloth"))
FAMILIES["gyro"] = dict(S.FAMILIES["gyro"], profile="expert-profile.bin",
                        models=models_of("gyro"))
FAMILIES["nvfp4"] = dict(S.FAMILIES["nvfp4"], profile="expert-profile.bin",
                         models=models_of("nvfp4"))

# The own-path q36 and the two .strata shapes with their own engine binaries.
FAMILIES["q36"] = dict(S.OWN_PATH_FAMILIES["qwen36"], models=list(S.OWN_PATH_FAMILIES["qwen36"]["models"]))
FAMILIES["hy4"] = dict(S.HY4_FAMILIES["hy4-flash"], models=list(S.HY4_FAMILIES["hy4-flash"]["models"]))
FAMILIES["inkling"] = dict(S.INKLING_FAMILIES["inkling-2-mini"],
                           models=list(S.INKLING_FAMILIES["inkling-2-mini"]["models"]))

KIND = {
    "qwen": "pack", "swift": "pack", "abliterated": "pack", "coder": "pack",
    "unsloth": "pack", "gyro": "pack", "nvfp4": "ready", "q36": "own",
    "hy4": "hy4", "inkling": "inkling",
}

# The GPU image encoder takes about 1.4 GB of VRAM; upstream puts it on cards of 24 GB or more
# (vision_reader's GPU_MIN_VRAM_GB) and reads pictures on the CPU below.
VISION_GPU_MIN_VRAM_MIB = 24064
VISION_MAX_TOKENS = {"gpu": 1024, "cpu": 300}
VISION_THREADS_CAP = 16


# The shard that carries the PLE table. setup scans the GGUF headers after the download
# (shard 2 for qwen and coder, shard 1 for swift; SC117's re-quant keeps Swift's split).  A
# dry run has no files to scan, so the hint drives the printed arguments; in a real run the
# files are on disk before the arguments are built and the scan's answer is authoritative.
PLE_SHARD = {"qwen": 2, "coder": 2, "swift": 1, "abliterated": 1}


def find_ple(family: str, gg: list) -> Path | None:
    """The shard holding per_layer_token_embd.weight, by scanning the headers when the files are
    there, by the published layout when they are not.  4-shard packs get no --ple-gguf: the
    engine finds its table in the shard set itself (upstream writes the flag for 2-shard packs
    only)."""
    if len(gg) > 2:
        return None
    present = [s for s in gg if s.exists()]
    if len(present) == len(gg):
        import gguf_reader as G
        for s in present:
            try:
                if any(t.name == "per_layer_token_embd.weight" for t in G.GGUFFile(s).tensors):
                    return s
            except (ValueError, struct.error):
                pass
        return None
    i = PLE_SHARD.get(family)
    return gg[i - 1] if i and 1 <= i <= len(gg) else None


def engine_bin(family: str) -> str:
    """The engine binary a family runs: strata-hy4 and strata-inkling are their own targets,
    installed beside strata by the flake's engine package."""
    if family == "hy4":
        return hy4_family.EXE
    if family == "inkling":
        return inkling_family.EXE
    return "strata"


def contexts_of(family: str) -> list:
    if family == "hy4":
        return list(hy4_family.CONTEXTS)
    if family == "inkling":
        return list(inkling_family.CONTEXTS)
    if family == "q36":
        return [c for c in CONTEXTS_ALL if c <= 262144]   # trained 262,144: no rope scaling on this path
    return CONTEXTS_ALL


def model_folder(family: str, fam: dict, model: str, data: Path) -> Path:
    """Where the family's files live, in upstream's own layouts: the GGUF packs in
    models/<tag><model>, Hy4-Flash and Inkling in their family folder, the ready packs in the
    repo-named folder."""
    if family == "hy4":
        return data / "models" / hy4_family.FAMILY
    if family == "inkling":
        return data / "models" / inkling_family.FAMILY
    if family == "nvfp4":
        return data / "models" / S.MODELS[model]["repo"].replace("/", "--")
    return data / "models" / (fam["tag"] + model)


def pack_dir(family: str, fam: dict, model: str, data: Path) -> Path:
    if family == "nvfp4":
        return S.ready_paths(model, data / "models")["pack"]
    return data / "packs" / (fam["tag"].lower() + model.lower())


def vision_pin(family: str, fam: dict, model: str, data: Path):
    """(url, file, size, sha) for the family's picture reader, from setup's own tables. None for
    hy4 and inkling: their Vision lives inside the family's .strata files, read by the engine."""
    if family in ("hy4", "inkling"):
        return None
    f = fam.get("vision_file", fam.get("mmproj"))
    base = fam.get("vision_hf") or fam.get("mmproj_hf")
    pin = fam.get("vision_pin")
    if pin is None:
        # the pin comes from setup's own table for the repo the picture reader is published in
        repo = fam.get("vision_repo")
        pin = S.REPO_PINS.get(repo, {}).get(f) if repo else None
    if pin is None and family == "q36":
        pin = S.QWEN36_FILES[f]
    if pin is None:
        pin = fam.get("mmproj_sha256")
    if pin is None:
        pin = S.ISTA_FILES[f]
    return (base + f, f, pin[0], pin[1])


def file_plan(family: str, fam: dict, model: str, vision: bool, folder: Path, data: Path) -> list:
    """(url, dst, what, sha, size) for every file the family needs, from upstream's own tables."""
    out = []
    if family in ("qwen", "swift", "abliterated", "coder"):
        # the layout is per repo, the way setup's own hf template says: qwen and coder publish
        # in a per-quant folder, swift at the repo root. abliterated (SC117's re-quant, not in
        # upstream's FAMILIES) publishes in per-quant folders, the way its repo lays them out.
        up = S.FAMILIES.get(family)
        base = (S.model_hf(up, model) if up
                else f"https://huggingface.co/{SC117}/resolve/{ABLITERATED_REV}/{model}/"
                if model not in fam.get("root_models", [])
                else f"https://huggingface.co/{SC117}/resolve/{ABLITERATED_REV}/")
        for i in range(1, fam["shards"] + 1):
            f = S.model_file(up, model, i) if up else fam["file"].format(q=model, i=i)
            pin = S.REPO_PINS.get(fam["repo"], {}).get(f)
            out.append((base + f, folder / f, f"{fam['title']} {model} shard {i}",
                        pin[1] if pin else None, pin[0] if pin else 0))
        if vision:
            url, f, size, sha = vision_pin(family, fam, model, data)
            out.append((url, data / "models" / f, f"picture reader {f}", sha, size))
    elif family in ("unsloth", "gyro"):
        base = S.model_hf(fam, model)
        for i in range(1, S.model_shards(fam, model) + 1):
            f = S.model_file(fam, model, i)
            size, sha = S.fam_pin(fam, Path(f))
            out.append((base + f, folder / f, f"{fam['title']} {model} shard {i}", sha, size))
        if vision:
            url, f, size, sha = vision_pin(family, fam, model, data)
            out.append((url, data / "models" / f, f"picture reader {f}", sha, size))
    elif family == "nvfp4":
        # ready_files yields (role, repo, rev, path, size, sha); ready_where wants the tuple
        # itself (it lays the file at models_dir/<repo with -- >/<path>)
        for e in S.ready_files(model):
            if e[0] == "src":
                continue                       # a prepare's source checkpoint belongs to the prepare step
            out.append((f"https://huggingface.co/{e[1]}/resolve/{e[2]}/{e[3]}",
                        S.ready_where(data / "models", e),
                        f"{fam['title']} {e[3]}", e[5], e[4]))
        if vision:
            url, f, size, sha = vision_pin(family, fam, model, data)
            out.append((url, data / "models" / f, f"picture reader {f}", sha, size))
    elif family == "q36":
        f = fam["file"].format(q=model)
        size, sha = S.QWEN36_FILES[f]
        out.append((fam["hf"] + f, folder / f, f"{fam['title']} {model}", sha, size))
        if vision:
            url, f, size, sha = vision_pin(family, fam, model, data)
            out.append((url, folder / f, f"picture reader {f}", sha, size))
    elif family == "hy4":
        src = hy4_family.sources()[0]
        for name in hy4_family.files_of(model, vision):
            urls = hy4_family.urls_for(name, src)
            size, sha = hy4_family.FILES[name]
            out.append((urls[0], folder / name, f"{fam['title']} {name}", sha, size))
            if len(urls) > 1:                # big builds serve as parts; hy4_fetch joins them
                for j, u in enumerate(urls[1:], 2):
                    out.append((u, folder / f"{name}.part-{j:02d}", f"{name} part {j}", None, 0))
    elif family == "inkling":
        for name in inkling_family.files_of(model, vision, dspark=False):
            src = inkling_family.sources_for(name)[0]
            pins = inkling_family.pins_for(name)
            size, sha = pins[0] if pins else (0, None)
            out.append((inkling_family.urls_for(name, src)[0], folder / name,
                        f"{fam['title']} {name}", sha, size))
    return out


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None):
    say(f"[strata-prefetch] ERROR: {msg}")
    if hint:
        say(f"  {hint}")
    sys.exit(1)


def ok(msg):
    say(f"[strata-prefetch] ok: {msg}")


def done(path: Path) -> bool:
    return path.with_name(path.name + ".done").exists()


def token_headers(token) -> dict:
    headers = {"User-Agent": "strata-nix"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def verify_file(dst: Path, sha: str, size: int, own: bool) -> None:
    """A downloaded file against its published pin. With a size, through setup's own
    verify_sha256 (which checks size + sha and writes the finish mark); with the sha alone
    (SC117's picture reader: its size is not in upstream's tables), the sha is checked here and
    setup's mark is written. A mismatch deletes the file and stops."""
    if size:
        S.verify_sha256(dst, size, sha, own=own)
        return
    digest = sha256_of(dst)
    if digest != sha:
        dst.unlink()
        fail(f"{dst.name}: the sha256 is {digest}, not the published {sha}", "delete it and run again")
    S.mark(dst, f"sha256 {sha}")


def download(url: str, dst: Path, what: str, token) -> None:
    """Resumable download with a size check (a .done mark short-circuits). The SHA-256 check for
    pinned files runs after, through upstream's own verify_sha256, which writes the finish mark."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what} already present")
        return
    headers = token_headers(token)
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=headers)
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check the network and run again (everything else is kept)")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:
        S.mark(dst)
        ok(f"{what} already present")
        return
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 10:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            say(f"  download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}", "check the network and run again (the download resumes)")
    part.replace(dst)
    ok(f"{what} downloaded")


def check_shard(shard: Path) -> None:
    """The shard is present and whole: its header must not name tensors running past the file's end."""
    import gguf_reader as G
    if not shard.exists():
        fail(f"missing {shard}")
    try:
        g = G.GGUFFile(shard)
    except (ValueError, struct.error) as e:
        fail(f"{shard.name} is not a whole GGUF shard ({e})", "delete it and run again")
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    if shard.stat().st_size < need:
        fail(f"{shard.name} is short: {shard.stat().st_size:,} of {need:,} bytes",
             "delete it and run again (or put a whole shard into --gguf-dir)")


def run_tool(script: str, *args, env=None) -> None:
    cmd = [sys.executable, str(TOOLS / script), *(str(a) for a in args)]
    say("  > " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        fail(f"{script} failed (exit {r.returncode})")


def mem_total_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    for cand in (found, "/run/current-system/sw/bin/nvidia-smi", "/usr/bin/nvidia-smi"):
        if cand and os.path.exists(cand):
            return cand
    return None


def vram_mib(gpu: str | None) -> int | None:
    """The card's VRAM as nvidia-smi reports it, in MiB: the --gpu card when one is named, else the
    card with the most (the engine's own pick). None when nvidia-smi cannot be asked: the PC reads
    as a unified-memory machine, as upstream treats an iGPU/UMA box (no VRAM/SSD caps to write)."""
    smi = nvidia_smi()
    if smi is None:
        return None
    cmd = [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    if gpu:
        cmd += ["-i", gpu.split(",")[0].strip()]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    values = [int(v) for v in r.stdout.splitlines() if v.strip().isdigit()]
    return max(values) if values else None


def resolve_vision(asked: str, gpu: str | None) -> str:
    """'auto' -> the GPU encoder on a card of 24 GB of VRAM or more, the CPU encoder below
    (upstream's rule); what was asked for otherwise, kept as it is."""
    if asked != "auto":
        mib = vram_mib(gpu)
        if asked == "gpu" and mib is not None and 0 < mib < VISION_GPU_MIN_VRAM_MIB:
            say(f"[strata-prefetch] images: the GPU encoder keeps ~1.4 GB of VRAM that a {mib / 1024:.0f} GB card "
                "would use for experts; kept as asked (--vision cpu reads pictures on the CPU instead)")
        return asked
    mib = vram_mib(gpu)
    if mib is None:
        say("[strata-prefetch] images: no nvidia-smi to ask the card's memory; the CPU encoder it is")
        return "cpu"
    mode = "gpu" if mib >= VISION_GPU_MIN_VRAM_MIB else "cpu"
    say(f"[strata-prefetch] images: the {mode} encoder (the card has {mib / 1024:.1f} GB of VRAM)")
    return mode


def vision_threads() -> int:
    return max(1, min((os.cpu_count() or 8) // 2, VISION_THREADS_CAP))


def build_args(a, fam, kind, folder, pack, rt, ram, vram, vision_on, low_ram, mode, data: Path) -> list:
    """The engine's command line, per family kind. The shapes are setup.py's: pack/ready follow
    strata_control/setup_config, q36/hy4/inkling follow install_own_path / install_hy4_flash /
    install_inkling2. A dry run never reaches the inkling drafter probe (network-free)."""
    ctx = a.context
    extra = a.extra_arg
    if kind in ("pack", "ready"):
        if kind == "ready":
            rp = S.ready_paths(a.model, data / "models")
            dense = str(rp["dense"])
            args = ["--pack", str(rp["pack"]), "--native", dense, "--native-dense-gguf", dense,
                    "--ple-gguf", str(rp["ple"])]
            if "embd" in rp:
                args += ["--embd-gguf", str(rp["embd"])]
            rt_arg = str(S.prepared_dir(a.model, data / "models") / "mtp") \
                if S.prepared_done(a.model, data / "models") else str(rt)
        else:
            gg = [p[1] for p in file_plan(a.family, fam, a.model, False, folder, data)
                  if p[1].suffix == ".gguf"]
            ple = find_ple(a.family, gg)
            args = ["--pack", str(pack), "--native", str(gg[0])]
            if len(gg) <= 2 and ple is not None:
                args += ["--ple-gguf", str(ple)]
            rt_arg = str(rt)
        args += ["--expert-profile", str(SHARE / "data" / fam.get("profile", "expert-profile.bin")),
                 "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
                 "--mtp", rt_arg, "--max-context", str(ctx)]
        # setup's KV rule: from 8K up the cache runs 8-bit; from 64K it streams from RAM when it fits
        kv = "int8" if ctx > 8192 else "fp16"
        if ctx > 8192:
            args += ["--kv", kv]
        kv_gb = S.kv_streaming_ram_gb(ctx, kv)
        stream = ctx >= 65536 and ram >= S.kv_streaming_need_gb(a.model, ctx, kv)
        if stream:
            args += ["--kv-resident", "32768"]
        if S.MODELS[a.model].get("budget"):           # Unsloth / NVFP4: the RAM-budget mode
            rb = S.resident_budget_gib(a.model, ram, kv_gb if stream else 0.0, False)
            if rb <= 0:
                args += ["--ssd-experts"]
                say(f"[strata-prefetch] note: {a.model}'s cold experts stream from the SSD on this PC")
            else:
                args += ["--resident-budget-gib", str(rb)]
        elif low_ram == "on":
            args += ["--mmap-experts"]
        if vision_on:
            args.append("--vision")
            if mode == "gpu" and not any(x == "--vram-reserve-mib" for x in extra):
                args += ["--vram-reserve-mib", "700"]
        return args + extra
    if kind == "own":
        gguf = str([p[1] for p in file_plan(a.family, fam, a.model, vision_on, folder, data)
                    if p[1].suffix == ".gguf"][0])
        args = ["--native", gguf, "--max-context", str(ctx), "--prefill", "auto", "--mtp", str(fam["mtp"])]
        if a.kv and a.kv in ("fp32", "fp16", "fp8", "fp4"):
            args += ["--kv", a.kv]
        if vision_on:
            args.append("--vision")
        return args + extra
    if kind == "hy4":
        args = ["--native", str(folder / hy4_family.build_file(a.model)),
                "--engram", str(folder / hy4_family.ENGRAM_FILE),
                "--max-context", str(ctx), "--session-reuse", "1"]
        if vision_on:
            args += ["--vision", str(folder / hy4_family.VISION_FILE)]
        gpus = len([g for g in (a.gpu or "").split(",") if g.strip()])
        unified = vram is None
        if gpus >= 2:
            args += ["--layer-split", "auto"]
        if not unified:
            placed = hy4_fit.place(a.model, vram / 1024, ram, unified, ctx, vision_on, extra_vram_gib=0.0)
            args += ["--vram-experts-gb", f"{placed['vram_gb']:.2f}",
                     "--ram-experts-gb", f"{placed['ram_gb']:.2f}"]
            if placed.get("ssd_gb", 0.0) > 1.0:
                say(f"[strata-prefetch] note: {a.model} spills ~{placed['ssd_gb']:.0f} GB of experts to the SSD; "
                    "a smaller build fits this PC better")
        return args + extra
    # inkling
    args = ["--native", str(folder / inkling_family.build_file(a.model)), "--max-context", str(ctx)]
    if a.kv and a.kv in ("fp16", "fp8", "k8v8", "k8v4"):
        args += ["--kv", a.kv]
    drafter_on = a.drafter != "off"
    if not drafter_on:
        args += ["--draft", "off"]
    if vision_on:
        args += ["--vision", str(folder / inkling_family.VISION_FILE)]
    if drafter_on and inkling_family.is_gguf_build(a.model):
        # the drafter file the engine can read (mtp.safetensors when the engine reads f32, else
        # Unsloth's Q8_0 copy); a check never downloads either, so the flag appears only when a
        # file is already in the data dir
        d = folder / inkling_family.MTP_FILE
        if not d.is_file():
            d = folder / inkling_family.DSPARK_GGUF_FILE
        if d.is_file():
            args += ["--dspark", str(d)]
    gpus = len([g for g in (a.gpu or "").split(",") if g.strip()])
    unified = vram is None
    if gpus >= 2:
        args += inkling_fit.multi_gpu_flags([{"index": i, "vram_gb": 0} for i in range(gpus)], windows=False)
    elif not unified:
        placed = inkling_fit.place(a.model, vram / 1024, ram, unified, ctx, vision_on,
                                   parallel=a.parallel or 1, dspark=drafter_on, backend="cuda")
        args += S.inkling_expert_flags(placed, backend="cuda")
        if placed.get("ssd_gb", 0.0) > 0.05:
            args += ["--ssd-experts"]
            say(f"[strata-prefetch] note: ~{placed['ssd_gb']:.1f} GB of experts stream from the SSD")
    return args + extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="swift", choices=sorted(FAMILIES))
    ap.add_argument("--model", default="IQ2_XS", help="the quant/build; each family has its own set")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gguf-dir", help="a directory holding the family's files already downloaded")
    ap.add_argument("--low-ram", choices=["on", "off"], help="default: automatic from the RAM and the model's need")
    ap.add_argument("--vision", choices=["off", "auto", "cpu", "gpu"], default="auto",
                    help="images: auto picks the GPU encoder from 24 GB of VRAM, the CPU one below; "
                         "Hy4-Flash and Inkling-2-Mini read pictures through their own Vision file")
    ap.add_argument("--vision-exe", help="the packaged strata-vision binary (needed for the GGUF families)")
    ap.add_argument("--kv", help="KV cache type: inkling fp16/fp8/k8v8/k8v4, q36 fp32/fp16/fp8/fp4")
    ap.add_argument("--drafter", choices=["dspark", "off"], help="Inkling-2-Mini GGUF builds: DSpark (default) or off")
    ap.add_argument("--parallel", type=int, metavar="N",
                    help="several requests at once; written into the config's 'parallel' key")
    ap.add_argument("--emit-config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be downloaded and built, then stop")
    ap.add_argument("--engine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--unload-secs", type=int, default=None)
    ap.add_argument("--gpu")
    ap.add_argument("--api-key-file")
    ap.add_argument("--hf-token-file")
    ap.add_argument("--sampling-json")
    ap.add_argument("--mcp-json")
    ap.add_argument("--extra-arg", action="append", default=[])
    a = ap.parse_args()

    fam = FAMILIES[a.family]
    kind = KIND[a.family]
    if a.model not in fam["models"]:
        fail(f"{a.family} has no {a.model} model", "choose one of: " + ", ".join(fam["models"]))
    if a.context not in contexts_of(a.family):
        fail(f"{a.family} does not use context {a.context}",
             "choose one of: " + ", ".join(str(c) for c in contexts_of(a.family)))

    data = Path(a.data_dir)
    pack = pack_dir(a.family, fam, a.model, data)
    mtp = data / "mtp"
    rt = mtp / "rt"
    folder = Path(a.gguf_dir) if a.gguf_dir else model_folder(a.family, fam, a.model, data)
    own = bool(a.gguf_dir)
    ram = mem_total_gb()
    vram = vram_mib(a.gpu)

    vision_on = a.vision != "off"
    mode = None
    vf = None
    if vision_on and kind in ("pack", "ready", "own"):
        mode = resolve_vision(a.vision, a.gpu)
        vf = vision_pin(a.family, fam, a.model, data)
        if vf is None:
            fail(f"no picture reader file is published for the {a.family} family",
                 'set vision = "off" (text only), or use a family that has one')
        if not a.vision_exe:
            fail("--vision needs --vision-exe (the packaged strata-vision binary)")
    if vision_on and kind in ("hy4", "inkling"):
        say(f"[strata-prefetch] images: on, read by the engine "
            f"({hy4_family.VISION_FILE if kind == 'hy4' else inkling_family.VISION_FILE})")

    token = Path(a.hf_token_file).read_text(encoding="utf-8").strip() if a.hf_token_file else None
    model_meta = S.MODELS.get(a.model, {})

    # low-RAM: GGUF packs map experts from experts.bin; the budget families (Unsloth, NVFP4) run
    # the RAM-budget mode when the RAM cannot hold every expert (setup's own rule).
    low_ram = a.low_ram or "auto"
    if low_ram == "auto":
        if model_meta.get("budget"):
            low_ram = "on" if ram < model_meta["arena_gb"] / 1.073741824 + S.UNSLOTH_RESIDENT_BESIDE_GB else "off"
        else:
            low_ram = "on" if ram < model_meta.get("ram_gb", 48) else "off"

    plan = file_plan(a.family, fam, a.model, vision_on, folder, data)
    args = build_args(a, fam, kind, folder, pack, rt, ram, vram, vision_on, low_ram, mode, data)

    steps = []
    # (in a real run the arguments are built again below, once the files are on disk)
    if kind == "pack":
        pack_args = " ".join(fam.get("pack_args", []))
        steps.append(f"pack: tools/iq_pack.py --gguf shard1 --out {pack} {pack_args}"
                     + (" + --experts-bin (low-RAM mode)" if low_ram == "on" else ""))
        steps.append(f"MTP draft layer: fetch ~5 GB of MTP tensors, pack (mtp_pack.py, mtp_rt.py) into {rt}")
    if kind == "ready":
        steps.append(f"ready pack: setup's ready_fetch downloads {S.MODELS[a.model]['repo']}'s pinned "
                     f"files into {folder}, size- and SHA-256-verified")
        if model_meta.get("prepare"):
            steps.append("prepare: tools/nvfp4_local_pack.py prepare (this PC turns NVIDIA's checkpoint "
                         "into a Strata pack once; needs torch/transformers/safetensors)")
        steps.append(f"MTP draft layer: a prepared pack carries its own draft layer; unprepared packs "
                     f"fetch ~5 GB into {rt}")
    if kind == "own":
        steps.append(f"tokenizer: tools/strata_tokenizer.py --gguf the model file --out {pack}")
    if kind in ("hy4", "inkling"):
        steps.append(f"tokenizer: Section 0 of the model file (Python 3.14 compression.zstd) into {pack}/tokenizer")
    if kind == "inkling" and inkling_family.is_gguf_build(a.model) and a.drafter != "off":
        steps.append(f"DSpark drafter: {inkling_family.MTP_FILE} (engine that reads it) or "
                     f"{inkling_family.DSPARK_GGUF_FILE} into {folder}")

    if a.dry_run:
        exe = str(Path(a.engine).parent / engine_bin(a.family)) if a.engine else engine_bin(a.family)
        say(f"[strata-prefetch] plan for {a.family}/{a.model} (context {a.context}, vision "
            + (str(mode) if mode else str(vision_on)) + f", engine {exe}, low-ram {low_ram}"
            + (f", parallel {a.parallel}" if a.parallel else "") + "):")
        for url, dst, what, sha, size in plan:
            say(f"  {what}\n    {url}\n    into {dst}" + (f"\n    sha256 {sha}" if sha else ""))
        if not plan:
            say("  (nothing to download)")
        for s in steps:
            say("  then: " + s)
        say("  engine args: " + " ".join(args))
        return 0

    # downloads: upstream's own fetchers for the families that have them (they download, verify
    # by SHA-256 and write the finish mark); the resumable downloader plus setup's verify_sha256
    # for the GGUF packs.
    for url, dst, what, sha, size in plan:
        if ".part-" in dst.name:
            continue                           # the fetcher downloads and joins parts itself
        if kind in ("hy4", "inkling"):
            # upstream's fetcher: a file already in the folder is verified against its pin and
            # kept; with --gguf-dir (own) a missing file is refused, not downloaded.
            shim = type("A", (), {"gguf_dir": None})()
            fetch = S.hy4_fetch if kind == "hy4" else S.inkling_fetch
            fetch(shim, dst.name, folder, own)
            continue
        if kind == "ready":
            continue                           # ready_fetch downloads the whole set below
        marked = S.mark_text(dst) or ""
        if dst.exists() and (sha and f"sha256 {sha}" in marked):
            ok(f"{what} already present")
            continue
        if dst.exists() and done(dst) and not sha:
            ok(f"{what} already present")
            continue
        if own:
            # --gguf-dir: the file was supplied here and is not marked yet. Check it against
            # the pin, write the mark; nothing is downloaded, a missing file stops.
            if not dst.exists():
                fail(f"{what} is not in {folder}", "--gguf-dir must hold the family's files")
            verify_file(dst, sha, size, own) if sha else S.mark(dst)
            ok(f"{what} verified from {folder}")
            continue
        download(url, dst, what, token)
        if sha:
            verify_file(dst, sha, size, own)
        else:
            S.mark(dst)
        ok(f"{what} verified")
    if kind == "ready":
        S.ready_fetch(a.model, data / "models")
        ok("ready pack files present and SHA-256 verified")

    # the files are on disk: build the arguments again, the header scan is authoritative now
    args = build_args(a, fam, kind, folder, pack, rt, ram, vram, vision_on, low_ram, mode, data)

    # pack, draft layer, drafter, tokenizer
    if kind == "pack":
        gg = [p[1] for p in plan if p[1].suffix == ".gguf"]
        for s in gg:
            check_shard(s)
        shard1 = gg[0]
        if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
            say("[strata-prefetch] building the pack (iq_pack.py, a few minutes) ...")
            run_tool("iq_pack.py", "--gguf", shard1, "--out", pack, *fam.get("pack_args", []))
            ok(f"pack built: {pack}")
        else:
            ok(f"pack already built: {pack}")
        if low_ram == "on" and not (pack / "experts.bin").exists():
            say("[strata-prefetch] writing the experts.bin for the low-RAM mode (a few minutes) ...")
            run_tool("iq_pack.py", "--gguf", shard1, "--out", pack, "--experts-bin", *fam.get("pack_args", []))
        if not (rt / "experts.bin").exists():
            say("[strata-prefetch] fetching and packing the MTP draft layer (~5 GB, once) ...")
            run_tool("mtp_fetch.py", "fetch", "--out", mtp)
            run_tool("mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", mtp / "mtp-q2_0.gguf")
            run_tool("mtp_rt.py", "--gguf", mtp / "mtp-q2_0.gguf", "--out", rt)
        if not (rt / "draft_vocab.bin").exists():
            shutil.copyfile(SHARE / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
        ok(f"MTP ready: {rt}")
    elif kind == "ready":
        if model_meta.get("prepare") and not S.prepared_done(a.model, data / "models"):
            if importlib.util.find_spec("torch") is None:
                fail("preparing NVIDIA's FP4-Local checkpoint needs PyTorch, transformers and safetensors, "
                     "which this NixOS build of the app does not carry",
                     "pick the GPTQ or NVIDIA NVFP4 pack (no prepare needed), or run the prepare from upstream's source setup")
            S.ensure_nvfp4_local_packages()
            src = next(S.ready_where(data / "models", e) for e in S.ready_files(a.model) if e[0] == "src")
            # ready_where wants the tuple intact, e[3] is the file name inside the repo folder
            run_tool("nvfp4_local_pack.py", "prepare", "--model", str(src),
                     "--out", str(S.prepared_dir(a.model, data / "models")))
            ok("NVFP4 pack prepared from NVIDIA's checkpoint")
        if not S.prepared_done(a.model, data / "models") and not (rt / "experts.bin").exists():
            say("[strata-prefetch] fetching and packing the MTP draft layer (~5 GB, once) ...")
            run_tool("mtp_fetch.py", "fetch", "--out", mtp)
            run_tool("mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", mtp / "mtp-q2_0.gguf")
            run_tool("mtp_rt.py", "--gguf", mtp / "mtp-q2_0.gguf", "--out", rt)
            if not (rt / "draft_vocab.bin").exists():
                shutil.copyfile(SHARE / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
            ok(f"MTP ready: {rt}")
    elif kind == "own":
        gguf = next(p[1] for p in plan if p[1].suffix == ".gguf")
        check_shard(gguf)
        if not (pack / "tokenizer" / "vocab.json").exists():
            run_tool("strata_tokenizer.py", "--gguf", gguf, "--out", pack)
        ok(f"tokenizer: {pack / 'tokenizer'}")
    elif kind == "hy4":
        shim = type("A", (), {"gguf_dir": None})()
        engine = Path(Path(a.engine).parent / hy4_family.EXE) if a.engine else None
        out = S.hy4_tokenizer_dir(folder / hy4_family.build_file(a.model), pack / "tokenizer",
                                  engine, shim, folder, own, data=data)
        if out is None:
            fail("the tokenizer cannot be built from the model file",
                 "run again once the engine is present (strata-hy4 --export-tokenizer)")
        ok(f"tokenizer: {out}")
    elif kind == "inkling":
        shim = type("A", (), {"gguf_dir": None})()
        engine = Path(Path(a.engine).parent / inkling_family.EXE) if a.engine else None
        model_file = folder / inkling_family.build_file(a.model)
        if inkling_family.is_gguf_build(a.model) and a.drafter != "off":
            # the probe reads the engine's --info line (dspark_f32_widen=1): which drafter file the
            # installed engine can use. A check never reaches this path (dry-run stops above).
            reads_mtp = S.inkling_engine_reads_mtp(str(engine), str(model_file))
            got = S.inkling_fetch_drafter(shim, folder, own, reads_mtp)
            if got is not None:
                ok(f"drafter: DSpark ({got.name})")
        out = S.inkling_tokenizer_dir(shim, model_file, folder, pack / "tokenizer", engine, own, data=data)
        if out is None:
            fail("the tokenizer cannot be built yet", "run again once the engine is present")
        ok(f"tokenizer: {out}")

    prepared = {"args": args, "tokenizer": str(pack / "tokenizer"),
                "model_name": f"{fam['name']}-{a.model.lower()}", "context": a.context,
                "low_ram": low_ram == "on", "engine": engine_bin(a.family)}
    (data / f"prepared-{fam['tag'].lower()}{a.model.lower()}.json").write_text(
        json.dumps(prepared, indent=1), encoding="utf-8")
    ok(f"prepared: engine args for {a.family}/{a.model}, context {a.context}, vision "
       + str(mode if mode is not None else vision_on))

    if a.emit_config:
        if not a.engine:
            fail("--emit-config needs --engine")
        exe = str(Path(a.engine).parent / engine_bin(a.family))
        if not os.path.exists(exe):
            fail(f"the engine for this family is not installed: {exe}",
                 "the flake's engine package installs strata, strata-device, strata-hy4 and strata-inkling")
        cfg = {"exe": exe, "args": args, "tokenizer": str(pack / "tokenizer"),
               "model_name": f"{fam['name']}-{a.model.lower()}",
               "log": str(data / f"engine-{fam['tag'].lower()}{a.model.lower()}.log")}
        arch = fam.get("arch")
        if arch:
            cfg["arch"] = arch
        if kind == "hy4":
            cfg["drafting"] = hy4_family.DEFAULT_DRAFTER
        if kind == "inkling":
            cfg["drafting"] = "off" if a.drafter == "off" else inkling_family.DEFAULT_DRAFTER
        if a.host:
            cfg["host"] = a.host
        if a.gpu:
            cfg["gpu"] = a.gpu
        if a.api_key_file:
            cfg["api_key"] = Path(a.api_key_file).read_text(encoding="utf-8").strip()
        if a.sampling_json:
            cfg["sampling"] = json.loads(Path(a.sampling_json).read_text(encoding="utf-8"))
        if a.mcp_json:
            cfg["mcp_servers"] = json.loads(Path(a.mcp_json).read_text(encoding="utf-8"))
        if a.parallel is not None:
            cfg["parallel"] = a.parallel
        if a.unload_secs is not None:
            cfg["lazy_load"] = True
            cfg["idle_unload_s"] = a.unload_secs
        if vf is not None:
            mmproj = (folder / vf[1]) if a.family == "q36" else (data / "models" / vf[1])
            gguf_first = next(p[1] for p in plan if p[1].suffix == ".gguf" and p[1].name != vf[1])
            cfg["vision"] = {"exe": a.vision_exe, "mmproj": str(mmproj), "model": str(gguf_first),
                             "gpu": mode == "gpu", "max_tokens": VISION_MAX_TOKENS[mode]}
            if mode == "cpu":
                cfg["vision"]["threads"] = vision_threads()
        out = Path(a.emit_config)
        out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if a.api_key_file:
            os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
        ok(f"config written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

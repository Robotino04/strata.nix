# strata.nix

A NixOS service around [Strata](https://github.com/Niko1221/Strata), the engine
that runs **Qwen3.8-Flash-Next** (125B MoE, ~6B active) on a single NVIDIA GPU
and serves it over an OpenAI- and Anthropic-compatible API on localhost.

```
┌─────────────┐   OpenAI/Anthropic    ┌───────────────┐   stdin/stdout   ┌────────────────┐
│ your apps   │ ────  HTTP  API  ───▶ │  strata-serve │ ───────────────▶ │ strata engine  │
│ / web chat  │   http://127.0.0.1    │  (python)     │                  │ (CUDA, built   │
└─────────────┘                       └───────────────┘                  │  from source)  │
                                                                         └────────────────┘
        weights: downloaded once (resumable) into /var/lib/strata, not in the nix store
```

## Requirements

- NixOS on **x86_64-linux** with an NVIDIA RTX 20/30/40/50 GPU (12 GB VRAM or more)
  and the official driver (580+). About 64 GB of RAM for IQ2_XS and below.
- ~70 GB of free disk space for the model files (downloaded at first service start),
  plus ~0.9 GB for the picture reader when images are on (`vision` not `off`).
- The small families run on small machines: `inkling` (Inkling-2-Mini, 4 GB class) and
  `q36` (Qwen3.6-35B-A3B) fit a 12 GB card with 32 GB of RAM; `hy4` (Tencent Hy4-Flash)
  runs SQ-64 on 12 GB VRAM + 64 GB RAM. They run their own engine binaries
  (`strata-hy4`, `strata-inkling`), which ship in the engine package.

## Quick start

```nix
# flake.nix of your system
{
  inputs.strata-nix.url = "github:Yunor743/strata.nix";
  # ...
  outputs = { self, nixpkgs, strata-nix, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        strata-nix.nixosModules.default
        {
          services.strata.enable = true;
          # defaults: Swift 1.5, IQ2_XS, 32K context, API on 127.0.0.1:8080
        }
      ];
    };
  };
}
```

The service:

1. (first start) downloads the model shards from Hugging Face and builds the pack and
   the MTP draft layer - and, with `vision` not `off`, the picture reader's mmproj
   (~0.9 GB, sha256-checked). Interrupted? Start again, it picks up where it left off;
2. writes `/var/lib/strata/config.json` (engine arguments, the vision block if images
   are on, API key if any);
3. starts `strata-serve` on `http://127.0.0.1:8080`.

Loading pushes 35-55 GB into RAM and the machine can crawl for 1-3 minutes.
That is normal. API: `http://127.0.0.1:8080/v1/chat/completions` (OpenAI) and
`/v1/messages` (Anthropic). Chat in the browser: `http://127.0.0.1:8080/`.

## Options (`services.strata.*`)

| Option | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | Enable the service. |
| `package` | `strataPkgs.strata` | The strata-serve bundle (carries the engine). |
| `family` | `swift` | The model to run: `swift` (Swift 1.5: same accuracy, answers ~1.8x sooner), `abliterated` (Swift 1.5 uncensored, SC117), `qwen` (original), `coder` (half the experts, code-focused, ~32 GB RAM), `unsloth` (Unsloth's GGUF set, budget mode on small-RAM PCs), `gyro` (Gyro-S/Gyro-M), `nvfp4` (ready NVFP4 packs, no GGUF), `q36` (Qwen3.6-35B-A3B), `hy4` (Tencent Hy4-Flash), `inkling` (Inkling-2-Mini). `hy4` and `inkling` run their own engine binaries and read pictures through their own Vision file. |
| `model` | `IQ2_XS` | The quant, per family: `Q2_0` / `IQ2_XS` / `IQ3_XXS` / `IQ3_S` (qwen only) / `IQ1_M` (coder only); `UD-*` for unsloth, q36 and inkling's GGUF builds; `Gyro-S` / `Gyro-M`; `NVFP4` / `NVFP4-NVIDIA` / `NVFP4-QAD`; `SQ-64/96/128` for hy4, `SQ-20/15/10` plus ten `UD-*` for inkling. A quant published by one family is refused for another at eval time. |
| `context` | `32768` | 8192-1048576 tokens for the Flash-Next packs; q36 and inkling stop at 262144, hy4 at 131072. Past 262144 strata-prefetch adds yarn rope scaling (experimental upstream). |
| `kv` | `null` | KV cache type (the engine's `--kv`). Null: strata-prefetch applies upstream's rule (8-bit from 8K context up, the cache streams from RAM from 64K when it fits). `fp32`/`fp16`/`fp8`/`k8v8`/`k8v4` are for inkling and q36 (q36 accepts fp32). |
| `drafter` | `dspark` | Inkling GGUF builds only: fetch the DSpark drafter (default) or `off` (no drafting, the engine runs `--draft off`). Inkling's `SQ-*` builds carry their drafter in the file, so the option does nothing for them. |
| `host` / `port` | `127.0.0.1` / `8080` | Listen address. |
| `apiKeyFile` | `null` | Path to a file with the API key (systemd credential). Required when `host` is not loopback. |
| `hfTokenFile` | `null` | Hugging Face token file, only for gated repos. |
| `openFirewall` | `false` | Open the TCP port in the firewall. |
| `gpus` | `[ ]` | CUDA device indices; several = layer split. Empty: the card with the most VRAM. |
| `vision` | `auto` | Images: `auto` picks the GPU encoder (~1.4 GB of VRAM) on a card of 24 GB or more, the CPU encoder (a few seconds a picture) below, like upstream; `gpu` / `cpu` force one; `off` is text only. The `abliterated` family takes SC117's own picture reader (same name, published beside the text shards). |
| `parallel` | `null` | Several requests answered at once (2-8): each slot keeps its own session context in VRAM. Null: requests take turns. The server turns this into the engine's batch slots and lowers N itself when the card is short of VRAM. |
| `lowRam` | `null` | Map experts from disk instead of RAM copies (auto when RAM is short). |
| `sampling` | `{ }` | Sampling defaults (temperature, top_p, top_k, ...). |
| `mcpServers` | `{ }` | MCP servers for the web chat (Claude Desktop format; they run with the service's rights). |
| `extraEngineArgs` | `[ ]` | Extra engine arguments, e.g. `[ "--kv" "q4_0" ]`. |
| `mockEngine` | `false` | Scripted mock instead of the model (tests; no GPU, no download). |

Example with the original model, the best quant and a LAN-reachable API:

```nix
services.strata = {
  enable = true;
  family = "qwen";
  model = "IQ3_XXS";
  context = 131072;
  host = "0.0.0.0";
  port = 8080;
  apiKeyFile = "/run/secrets/strata-api-key";
  sampling = { temperature = 1.0; top_p = 0.95; top_k = 20; };
};
```

The small model on a small machine (runs its own engine, its own Vision file):

```nix
services.strata = {
  enable = true;
  family = "inkling";
  model = "SQ-20";        # or UD-Q4_K_XL, which takes the DSpark drafter step
  context = 131072;
};
```

## Packages

- `strata`, the server bundle: `strata-serve` (the API), `strata-chat` (the terminal
  chat), `strata-cli` / `strata-monitor` (upstream's `strata` CLI and its dashboard)
  and `strata-prefetch` (the download/prepare tool, usable standalone;
  `strata-prefetch --help`).
- `strata-engine`, the CUDA inference binary built from source
  (`strata-engine.override { cudaArch = "120"; }` for an RTX 50 card; default `89` = RTX 40).
- `strata-vision`, Strata's own image encoder (`terminal/tools/vision`: its Qwen
  vision tower, no llama.cpp in the binary), started by the server only while it reads
  a picture (`strata-vision.override { cudaArch = "120"; }` likewise). `hy4` and `inkling`
  do not use it; their engines read their own Vision file.
- `strata-engine` installs four binaries: `strata` (the Flash-Next packs), `strata-device`,
  `strata-hy4` (Hy4-Flash, with its engram and session reuse) and `strata-inkling`
  (Inkling-2-Mini, with DSpark drafting and multi-GPU layer split). The serve wrappers pick
  the right one per family.
- The model weights are **not** packaged: they are downloaded at runtime into the
  service's state directory (the nix store would not survive a 70 GB model well).

## How the engine is built

Strata vendors everything it builds against (`engine/third_party/strata-ggml`,
`strata-gguf-py`, `strata-llama`), so no llama.cpp source enters the build.
`strata-engine` compiles the pinned Strata source with nixpkgs' CUDA toolchain
(`backendStdenv`, CUDA 13.0: 12.9's `cicc` crashes on Strata's `native_mmvq.cu`, and
13.2.x miscompiles `sm_120`). The flake input `llama-pinned` is the pin witness: it
must be the commit Strata's `engine/third_party/ggml/VERSION.txt` names, so a vendor
bump is visible in the lock file. If Strata bumps that pin, evaluation fails with the
new commit and an instruction to update the `llama-pinned` input.

The flake input `strata` tracks upstream `main`. The lock file decides what you
actually run. `nix flake lock --update-input strata` pulls new releases
(and new engine features). Read what lands in the lock before you run it.

## Testing

- `nix flake check`: six checks, all download-free — a serve smoke test (mock engine: HTTP
  endpoints without a GPU), a NixOS VM test of the systemd service, module checks for the
  vision, parallel and extra-args wiring, and a prefetch vision check. Upstream packaging
  (`terminal/setup.py`, `strata-prefetch`) is the source of truth for the model tables:
  strata-prefetch imports the packaged setup.py, so upstream's pins and rules land here
  without a mirror to keep in sync.
- `nix build .#strata-engine`: the real compile (20-40 minutes, once, cached after).
- `nix build .#strata-vision`: the image encoder's compile (Strata's own vision tower,
  its CUDA kernels).
- `strata-prefetch --dry-run` prints what would be downloaded and built, without doing it.

## Licenses

- This flake: MIT.
- Strata (engine, server, tools): MIT, upstream <https://github.com/Niko1221/Strata>.
- The model weights are not part of this flake; each model's license applies
  (Qwen Community 1.0; Swift Open License 1.0; see the models' Hugging Face pages).

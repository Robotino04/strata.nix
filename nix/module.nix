{ strataPkgs }:
{ pkgs, config, lib, ... }:
let
  cfg = config.services.strata;

  # The families that read pictures through their own Vision file (the .strata builds): the
  # engine loads it, so no strata-vision is passed to them.
  strataVisionFamilies = [ "hy4" "inkling" ];

  # The quant each family publishes, mirrored from setup.py's MODELS table and the repos' trees
  # (the same lists strata-prefetch builds from the share).  The module refuses an unpublished
  # pair at evaluation time, before the service starts downloading.
  familyModels = {
    qwen = [ "Q2_0" "IQ2_XS" "IQ3_XXS" "IQ3_S" ];
    swift = [ "Q2_0" "IQ2_XS" "IQ3_XXS" "IQ3_S" ];
    abliterated = [ "Q2_0" "IQ2_XS" "IQ3_XXS" "IQ3_S" ];
    # the coder repo ships IQ1_M only
    coder = [ "IQ1_M" ];
    unsloth = [ "UD-Q2_0" "UD-IQ2_XS" "UD-Q2_K_XL" "UD-IQ3_XXS" "UD-Q3_K_XL" "UD-IQ4_XS"
      "UD-IQ4_XS-v31" "UD-Q4_K_XL" "UD-Q5_K_XL" "UD-Q6_K_XL" "Q8_0" "UD-IQ1_M" ];
    gyro = [ "Gyro-S" "Gyro-M" ];
    nvfp4 = [ "NVFP4" "NVFP4-NVIDIA" "NVFP4-QAD" ];
    q36 = [ "UD-Q4_K_XL" "UD-Q6_K_XL" "Q8_0" ];
    hy4 = [ "SQ-64" "SQ-96" "SQ-128" ];
    inkling = [ "SQ-20" "SQ-15" "SQ-10" "UD-Q8_K_XL" "UD-Q4_K_XL" "UD-Q4_K_M" "UD-IQ4_XS"
      "UD-Q3_K_XL" "UD-IQ3_XXS" "UD-Q2_K_XL" "UD-IQ2_XXS" "UD-IQ1_M" "UD-IQ1_S" ];
  };

  # The contexts each family uses: the Flash-Next packs keep the full 10; q36 stops at its
  # trained 262144 (no rope scaling on that path), Hy4-Flash at 131072, Inkling-2-Mini at 262144.
  familyContexts = {
    qwen = [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
    swift = [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
    abliterated = [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
    coder = [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
    unsloth = [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
    gyro = [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
    nvfp4 = [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
    q36 = [ 8192 32768 65536 131072 204800 262144 ];
    hy4 = [ 8192 32768 65536 131072 ];
    inkling = [ 8192 32768 65536 131072 204800 262144 ];
  };
in
{
  options.services.strata = {
    enable = lib.mkEnableOption "Strata, a local OpenAI/Anthropic-compatible API for Qwen3.8-Flash-Next (125B MoE)";

    package = lib.mkPackageOption strataPkgs "strata" { };

    family = lib.mkOption {
      type = lib.types.enum [ "qwen" "swift" "abliterated" "coder" "unsloth" "gyro" "nvfp4" "q36" "hy4" "inkling" ];
      default = "swift";
      description = ''
        Which model: `swift` (Swift 1.5, answers ~1.8x sooner for the same accuracy),
        `abliterated` (Swift 1.5 uncensored, SC117's re-quant; Q2_0/IQ2_XS/IQ3_XXS),
        `qwen` (the original Qwen3.8-Flash-Next) or `coder` (half the experts kept, code-focused);
        `unsloth` (Unsloth's GGUF set, budget mode on small-RAM PCs) and `gyro` (Gyro-S/Gyro-M);
        `nvfp4` (ready NVFP4 packs, no GGUF), `q36` (Qwen3.6-35B-A3B, the own-path route);
        `hy4` (Tencent Hy4-Flash) and `inkling` (Inkling-2-Mini) run their own engine binaries
        (strata-hy4, strata-inkling) and read pictures through their own Vision file.
      '';
    };

    model = lib.mkOption {
      type = lib.types.enum [
        "Q2_0" "IQ2_XS" "IQ3_XXS" "IQ3_S" "IQ1_M"
        "UD-Q2_0" "UD-IQ2_XS" "UD-Q2_K_XL" "UD-IQ3_XXS" "UD-Q3_K_XL" "UD-IQ4_XS" "UD-IQ4_XS-v31"
        "UD-Q4_K_XL" "UD-Q5_K_XL" "UD-Q6_K_XL" "UD-Q8_K_XL" "Q8_0" "UD-IQ1_M" "UD-IQ1_S" "UD-Q4_K_M"
        "UD-IQ2_XXS"
        "Gyro-S" "Gyro-M"
        "NVFP4" "NVFP4-NVIDIA" "NVFP4-QAD"
        "SQ-64" "SQ-96" "SQ-128" "SQ-20" "SQ-15" "SQ-10"
      ];
      default = "IQ2_XS";
      description = ''
        Quantization size, per family: Q2_0 (fastest), IQ2_XS (recommended), IQ3_XXS (better,
        slower), IQ3_S (best: qwen, swift, abliterated), IQ1_M (coder, the only quant its repo
        ships). The `UD-*` names are Unsloth's sets (unsloth, and the GGUF builds of inkling and
        q36); Gyro-S/Gyro-M are gyro's; the NVFP4 names are nvfp4's ready packs; SQ-64/96/128 are
        Hy4-Flash's builds and SQ-20/15/10 Inkling-2-Mini's. A quant published by one family is
        refused for another.
      '';
    };

    kv = lib.mkOption {
      type = lib.types.nullOr (lib.types.enum [ "fp32" "fp16" "fp8" "k8v8" "k8v4" ]);
      default = null;
      description = ''
        The KV cache type. Null (the default): strata-prefetch writes upstream's own rule (8-bit
        from 8K context up, the cache streams from RAM from 64K when it fits). fp8/k8v8/k8v4 and
        fp16 are for inkling and q36 (q36 accepts fp32), and reach the engine as --kv.
      '';
    };

    drafter = lib.mkOption {
      type = lib.types.enum [ "dspark" "off" ];
      default = "dspark";
      description = ''
        Inkling-2-Mini GGUF builds only: the DSpark drafter (the default; the engine's file is
        fetched when the engine can read it) or off (drafting stops; the engine runs --draft off).
      '';
    };

    context = lib.mkOption {
      type = lib.types.enum [ 8192 32768 65536 131072 204800 262144 393216 524288 786432 1048576 ];
      default = 32768;
      description = ''
        Context window in tokens. From 64K on, the KV cache streams from RAM.
        Past 262144 (the trained length) strata-prefetch turns yarn on by itself
        and derives the factor; extraEngineArgs can override.
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Listen address. Use 0.0.0.0 to serve the LAN (requires apiKeyFile).";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
    };

    apiKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding the API key (read through a systemd credential, never in the unit).";
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding a Hugging Face token (only for gated model repos).";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };

    gpus = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = ''
        CUDA device numbers (as nvidia-smi numbers them, PCI order).
        Empty: the engine picks the card with the most VRAM.
        Several: layer split across the cards.
      '';
    };

    vision = lib.mkOption {
      type = lib.types.enum [ "off" "auto" "cpu" "gpu" ];
      default = "auto";
      description = ''
        Images (OpenAI image_url parts, Anthropic image blocks, the web chat's paperclip):
        `auto` picks the GPU encoder (about 1.4 GB of VRAM the engine reserves) on a card of 24 GB
        or more and the CPU encoder (a few seconds a picture, no VRAM) below, like upstream;
        `gpu` and `cpu` force one; `off` is text only.  With images on, strata-prefetch also
        fetches the picture reader file (~0.9 GB) for the family.
      '';
    };

    parallel = lib.mkOption {
      type = lib.types.nullOr (lib.types.ints.between 1 8);
      default = null;
      description = ''
        Several requests answered at once (2-8): each slot keeps its own session context in
        VRAM, so agents that run several sessions or several people on one host stop waiting
        for each other.  Null (the default): requests take turns, one at a time.  The server
        turns this into the engine's batch slots and lowers the number itself when the card
        is short of VRAM; N over the engine's limit is clamped, not refused.
      '';
    };

    unloadSec = lib.mkOption {
      type = lib.types.nullOr lib.types.number;
      default = null;
      description = ''
        Unload the model after this many idle seconds (the engine is started again on the
        next request). Null (the default): the model stays loaded while the service runs.
      '';
    };

    lowRam = lib.mkOption {
      type = lib.types.nullOr lib.types.bool;
      default = null;
      description = ''
        Map the experts from the pack's experts.bin instead of copying them into RAM
        (needs a big GPU to stay fast). Null: automatic from the model's RAM needs.
      '';
    };

    sampling = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      example = { temperature = 1.0; top_p = 0.95; top_k = 20; };
      description = "Sampling defaults for requests that leave the fields out.";
    };

    mcpServers = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = ''
        MCP servers for the web app's chat, in Claude Desktop's format.
        They run with this service's rights; only add ones you trust.
      '';
    };

    extraEngineArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra arguments appended to the engine command line (e.g. --kv q4_0).";
    };

    mockEngine = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Serve a scripted mock engine instead of the model (tests; no GPU or download).";
    };

    prefetchOnStart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Download/prepare missing model files before starting (resumable, skipped when done).";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.model != "IQ3_S" || cfg.family == "qwen";
        message = "services.strata: IQ3_S is only published for family = qwen.";
      }
      {
        assertion = cfg.model != "IQ1_M" || cfg.family == "coder";
        message = "services.strata: IQ1_M is only published for family = coder.";
      }
      {
        assertion = cfg.model != "IQ2_XS" || cfg.family != "coder";
        message = "services.strata: the coder family ships IQ1_M only; pick model = IQ1_M.";
      }
      {
        assertion = builtins.elem cfg.model (familyModels.${cfg.family} or [ ]);
        message = "services.strata: ${cfg.model} is not published for family = ${cfg.family}: "
          + "choose one of ${lib.concatStringsSep ", " (familyModels.${cfg.family} or [ ])}.";
      }
      {
        assertion = builtins.elem cfg.context (familyContexts.${cfg.family} or [ ]);
        message = "services.strata: family = ${cfg.family} does not use context ${toString cfg.context}: "
          + "choose one of ${lib.concatStringsSep ", " (map toString (familyContexts.${cfg.family} or [ ]))}.";
      }
      {
        assertion = cfg.host == "127.0.0.1" || cfg.host == "localhost" || cfg.apiKeyFile != null;
        message = "services.strata: exposing the API beyond loopback requires apiKeyFile.";
      }
      {
        assertion = !cfg.mockEngine || !cfg.prefetchOnStart;
        message = "services.strata: mockEngine requires prefetchOnStart = false.";
      }
      {
        assertion = !cfg.mockEngine || cfg.vision == "off";
        message = "services.strata: mockEngine serves text only; set vision = \"off\" with it.";
      }
      {
        assertion = cfg.mockEngine || cfg.vision == "off" || builtins.elem cfg.family strataVisionFamilies
          || (cfg.package ? passthru.vision);
        message = "services.strata: with vision on, package must be the strata-serve bundle (it carries the vision encoder under passthru.vision).";
      }
      {
        assertion = cfg.mockEngine || (cfg.package ? passthru.engine);
        message = "services.strata: package must be the strata-serve bundle (it carries the engine under passthru.engine).";
      }
    ];

    systemd.services.strata = {
      description = "Strata: Qwen3.8-Flash-Next (125B MoE) local inference API";
      documentation = [ "https://github.com/Niko1221/Strata" ];
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ cfg.package ];

      environment = lib.optionalAttrs (cfg.gpus != [ ]) {
        CUDA_DEVICE_ORDER = "PCI_BUS_ID";
        CUDA_VISIBLE_DEVICES = lib.concatStringsSep "," cfg.gpus;
      };

      preStart = lib.mkIf (cfg.prefetchOnStart && !cfg.mockEngine) (toString (
        pkgs.writeShellScript "strata-prestart" (''
          set -euo pipefail
          state="$STATE_DIRECTORY"
          exec ${cfg.package}/bin/strata-prefetch \
            --family ${lib.escapeShellArg cfg.family} \
            --model ${lib.escapeShellArg cfg.model} \
            --context ${toString cfg.context} \
            --data-dir "$state" \
        '' + lib.optionalString (cfg.lowRam != null) ''
            --low-ram ${if cfg.lowRam then "on" else "off"} \
        '' + ''
            --vision ${cfg.vision} \
        '' + lib.optionalString (cfg.vision != "off" && !builtins.elem cfg.family strataVisionFamilies) ''
            --vision-exe ${lib.escapeShellArg (toString cfg.package.passthru.vision)}/bin/strata-vision \
        '' + lib.optionalString (cfg.parallel != null) ''
            --parallel ${toString cfg.parallel} \
        '' + lib.optionalString (cfg.kv != null) ''
            --kv ${cfg.kv} \
        '' + lib.optionalString (cfg.family == "inkling") ''
            --drafter ${cfg.drafter} \
        '' + lib.optionalString (cfg.unloadSec != null) ''
            --unload-secs ${toString cfg.unloadSec} \
        '' + lib.optionalString (cfg.hfTokenFile != null) ''
            --hf-token-file "/run/credentials/strata.service/hf-token" \
        '' + ''
            --emit-config "$state/config.json" \
            --engine ${lib.escapeShellArg (toString cfg.package.passthru.engine)}/bin/strata \
            --host ${lib.escapeShellArg cfg.host} \
        '' + lib.optionalString (cfg.apiKeyFile != null) ''
            --api-key-file "/run/credentials/strata.service/api-key" \
        '' + lib.optionalString (cfg.gpus != []) ''
            --gpu ${lib.escapeShellArg (lib.concatStringsSep "," cfg.gpus)} \
        '' + lib.optionalString (cfg.sampling != { }) ''
            --sampling-json ${pkgs.writeText "strata-sampling.json" (builtins.toJSON cfg.sampling)} \
        '' + lib.optionalString (cfg.mcpServers != { }) ''
            --mcp-json ${pkgs.writeText "strata-mcp.json" (builtins.toJSON cfg.mcpServers)} \
        '' + lib.concatMapStrings (x: ''--extra-arg=${lib.escapeShellArg x} '') cfg.extraEngineArgs
        )
      ));

      serviceConfig = {
        Type = "simple";
        DynamicUser = true;
        StateDirectory = "strata";
        StateDirectoryMode = "0750";
        # First start: preStart downloads ~70 GB and builds the pack - the default
        # 90 s TimeoutStartSec kills it mid-download (the service is resumable, but
        # converge it in one pass). Applies to the start, incl. ExecStartPre.
        TimeoutStartSec = "0";
        WorkingDirectory = "%S/strata";
        UMask = "0077";
        Restart = "on-failure";
        RestartSec = 10;
        LoadCredential =
          (lib.optional (cfg.apiKeyFile != null) "api-key:${cfg.apiKeyFile}")
          ++ (lib.optional (cfg.hfTokenFile != null) "hf-token:${cfg.hfTokenFile}");
        ExecStart =
          if cfg.mockEngine
          then
            "${cfg.package}/bin/strata-serve --engine mock --port ${toString cfg.port} " +
            "--host ${lib.escapeShellArg cfg.host}"
          else
            "${cfg.package}/bin/strata-serve --engine strata --config %S/strata/config.json " +
            "--port ${toString cfg.port}";
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_UNIX" "AF_INET" "AF_INET6" ];
        LockPersonality = true;
        MemoryDenyWriteExecute = false;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}

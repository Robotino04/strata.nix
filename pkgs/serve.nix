{
  lib,
  stdenvNoCC,
  makeWrapper,
  python3,
  strataSrc,
  engine,
  vision ? null,
  version,
  pinChecked ? true,
}:
assert pinChecked;
let
  # terminal/requirements.txt's runtime set; textual is requirements-monitor.txt (the
  # dashboard's own pin — the server imports none of it).
  py = python3.withPackages (ps:
    with ps; [
      numpy
      jinja2
      regex
      pyyaml
      tqdm
      requests
      pillow
      psutil
      textual
    ]);

  # tools/_paths.py finds the vendored gguf-py at <terminal>/../engine/third_party/strata-gguf-py,
  # so the share keeps that one engine/ directory and STRATA_GGUF_PY needs setting for nothing.
  share = stdenvNoCC.mkDerivation {
    pname = "strata-share";
    inherit version;
    src = strataSrc;
    phases = [ "installPhase" ];
    installPhase = ''
      mkdir -p $out
      cp -rL ${strataSrc}/terminal $out/terminal
      chmod -R u+w $out/terminal
      rm -rf $out/terminal/bench $out/terminal/ref $out/terminal/crates $out/terminal/tests
      mkdir -p $out/engine/third_party
      cp -rL ${strataSrc}/engine/third_party/strata-gguf-py $out/engine/third_party/strata-gguf-py
      chmod -R u+w $out/engine
    '';
  };
in
  stdenvNoCC.mkDerivation {
    pname = "strata";
    inherit version;
    dontUnpack = true;
    nativeBuildInputs = [ makeWrapper ];

    installPhase = ''
      runHook preInstall
      mkdir -p $out/bin $out/share
      cp -r ${share} $out/share/strata
      chmod -R u+w $out/share/strata
      # upstream has no prefetch: this flake's own download/prepare tool
      install -Dm644 ${./prefetch.py} $out/share/strata/prefetch.py
      # server.py inserts its own parents into sys.path, so it runs from the share
      makeWrapper ${py}/bin/python $out/bin/strata-serve \
        --add-flags "$out/share/strata/terminal/serve/server.py"
      makeWrapper ${py}/bin/python $out/bin/strata-chat \
        --add-flags "$out/share/strata/terminal/tools/chat.py"
      # the terminal CLI (strata serve/status/monitor/…) and the dashboard
      makeWrapper ${py}/bin/python $out/bin/strata-cli \
        --add-flags "$out/share/strata/terminal/strata.py"
      makeWrapper ${py}/bin/python $out/bin/strata-monitor \
        --add-flags "$out/share/strata/terminal/strata.py monitor"
      makeWrapper ${py}/bin/python $out/bin/strata-prefetch \
        --add-flags "$out/share/strata/prefetch.py" \
        --set STRATA_SHARE "$out/share/strata/terminal"
      runHook postInstall
    '';

    passthru = {
      engine = engine;
      vision = vision;
      inherit share py;
      prefetchFlags = { };
    };

    meta = {
      description = "Strata local inference server (OpenAI/Anthropic-compatible API for Qwen3.8-Flash-Next)";
      homepage = "https://github.com/Niko1221/Strata";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
      mainProgram = "strata-serve";
    };
  }

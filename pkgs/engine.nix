{
  lib,
  cmake,
  ninja,
  autoAddDriverRunpath,
  pkgs,
  src,
  llamaSrc,
  version,
  ggmlPin,
  pinChecked ? true,
  cudaArch ? "89",
  cudaVersion ? "12",
}:
assert pinChecked; let
  # Upstream supports two CUDA lines: CUDA 12.9 (VARIANT=cuda12, sm_75 .. sm_89) and
  # CUDA 13.0 (VARIANT=cuda13, sm_75 .. sm_120). CUDA 13.2.0/13.2.1 miscompile for
  # sm_120, so 13.0.88 is the pinned release. Select the matching package set.
  cudaPackages =
    if cudaVersion == "13"
    then pkgs.cudaPackages_13_0
    else if cudaVersion == "12"
    then pkgs.cudaPackages_12_9
    else throw "strata-engine: cudaVersion must be \"12\" (CUDA 12.9, sm_75..sm_89) or \"13\" (CUDA 13.0, sm_75..sm_120)";
in
  cudaPackages.backendStdenv.mkDerivation {
    pname = "strata-engine";
    inherit version;
    inherit src;
    strictDeps = true;

    nativeBuildInputs = [
      cmake
      ninja
      cudaPackages.cuda_nvcc
      autoAddDriverRunpath
    ];

    buildInputs = [cudaPackages.cudatoolkit];

    cmakeFlags = [
      "-DCMAKE_GENERATOR=Ninja"
      "-DCMAKE_BUILD_TYPE=Release"
      "-DSTRATA_ENABLE_CUDA=ON"
      "-DSTRATA_BUILD_TESTS=OFF"
      "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
      "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
      "-DSTRATA_GGML_DIR=${llamaSrc}"
      "-DGGML_NATIVE=OFF"
      "-DGGML_AVX=ON"
      "-DGGML_AVX2=ON"
      "-DGGML_FMA=ON"
      "-DGGML_F16C=ON"
      "-DGGML_BMI2=ON"
      "-DGGML_AVX512=ON"
    ];

    env = {
      PIN_ASSERT = toString pinChecked;
      GGML_PIN_ASSERT = ggmlPin;
      CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
      CUDAToolkit_INCLUDE_ROOT = cudaPackages.cudatoolkit;
      CUDAToolkit_LIBRARY_ROOT = cudaPackages.cudatoolkit;
    };

    buildFlags = ["--target" "strata" "--target" "strata-device"];

    postBuild = ''
      cmake -S ${src}/tools/vision \
            -B build-vision \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLAMA_DIR=${llamaSrc} \
            -DSTRATA_VISION_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=${cudaArch} \
            -DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc

      cmake --build build-vision --target strata-vision
    '';

    installPhase = ''
          runHook preInstall
          # Restrict the search to the cmake build dirs.  Upstream 0.1.55 ships a
          # shell launcher at `bin/strata` in the source tree, and `find
          # "$NIX_BUILD_TOP" -name strata` picks it up (the launch script, not the
          # compiled engine), so the engine was being swapped for the launcher,
          # which dies on "no usable Python found".
          #
          # Only the cmake build dirs contain compiled binaries, and their layout
          # is fixed; the launcher is not inside them.
          for d in build build-vision; do
            case $(find "$NIX_BUILD_TOP/$d" -maxdepth 4 -type f) in
              *strata*) ;;
              *) echo "missing cmake build dir: $d" >&2; exit 1 ;;
            esac
          done
          dev=$(find "$NIX_BUILD_TOP/build" -maxdepth 3 -type f -name strata-device | head -1)
          [[ -n "$dev" ]] && install -Dm755 "$dev" $out/bin/strata-device

          bin=$(find "$NIX_BUILD_TOP/build" -maxdepth 3 -type f -name strata | head -1)
          if [[ -z "$bin" ]]; then
            echo "strata binary not found under $NIX_BUILD_TOP/build" >&2
            exit 1
          fi
          install -Dm755 "$bin" $out/bin/strata

          vision=$(find "$NIX_BUILD_TOP/build-vision" -maxdepth 4 -type f -name strata-vision | head -1)
          if [[ -z "$vision" ]]; then
            echo "strata-vision binary not found under $NIX_BUILD_TOP/build-vision" >&2
            exit 1
          fi
          install -Dm755 "$vision" "$out/bin/strata-vision"

          cat > $out/bin/BUILD.json <<EOF
      {
        "version": "${version}",
        "archs": [${cudaArch}],
        "cuda": "${cudaPackages.cuda_nvcc.version}",
        "ptx": false,
        "source": "nix",
        "vision": "gpu"
      }
      EOF
          runHook postInstall
    '';

    passthru = {
      inherit cudaArch cudaVersion;
    };

    meta = {
      description = "Strata inference engine for Qwen3.8-Flash-Next (125B MoE) on a single NVIDIA GPU (sm_${cudaArch})";
      homepage = "https://github.com/Niko1221/Strata";
      license = lib.licenses.mit;
      platforms = ["x86_64-linux"];
      mainProgram = "strata";
    };
  }

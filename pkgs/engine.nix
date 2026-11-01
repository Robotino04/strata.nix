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
  cudaVersion ? "13",
}:
assert pinChecked; let
  # Upstream supports two CUDA lines: CUDA 12.9 (VARIANT=cuda12, sm_75 .. sm_89) and
  # CUDA 13.0 (VARIANT=cuda13, sm_75 .. sm_120). CUDA 12.9.86's cicc crashes
  # compiling src/kernels/cuda/native_mmvq.cu (signal 11), so CUDA 13.0.88 is the
  # default. 13.2.0/13.2.1 miscompile for sm_120, so 13.0.88 is the pinned release.
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
          # Upstream 0.1.55 ships a shell launcher at `bin/strata` in the
          # source tree. `find "$NIX_BUILD_TOP" -name strata` finds that launcher
          # script as well as the compiled engine, so `head -1` can install the
          # launcher and replace the engine, which dies at runtime with
          # "strata: no usable Python found".
          #
          # The cmake phase builds in `$sourceRoot/build` (the cmake build dir),
          # `postBuild` writes vision into `$sourceRoot/build/build-vision/bin`,
          # and `sourceRoot` = `source` (the basename of the input flake source).
          # Scope the search to that cmake build dir to pick compiled binaries
          # only.
          root="$NIX_BUILD_TOP/source"
          cmake_dir="$root/build"
          vision_dir="$cmake_dir/build-vision"
          if [[ ! -f "$cmake_dir/CMakeCache.txt" ]]; then
            echo "cmake build dir not found: $cmake_dir" >&2
            exit 1
          fi
          if [[ ! -f "$cmake_dir/strata" ]]; then
            echo "compiled strata binary not found: $cmake_dir/strata" >&2
            exit 1
          fi
          install -Dm755 "$cmake_dir/strata" $out/bin/strata
          if [[ -f "$cmake_dir/strata-device" ]]; then
            install -Dm755 "$cmake_dir/strata-device" $out/bin/strata-device
          fi
          if [[ ! -f "$vision_dir/bin/strata-vision" ]]; then
            echo "strata-vision binary not found under $vision_dir" >&2
            exit 1
          fi
          install -Dm755 "$vision_dir/bin/strata-vision" "$out/bin/strata-vision"

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

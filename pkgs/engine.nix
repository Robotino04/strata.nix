{
  lib,
  cmake,
  ninja,
  autoAddDriverRunpath,
  pkgs,
  zstd,
  src,
  version,
  ggmlPin,
  cudaArch ? "89",
  cudaVersion ? "13",
}:
let
  # 12.9.86's cicc crashes on native_mmvq.cu; 13.2.0/13.2.1 miscompile for sm_120. 13.0.88 works.
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
    sourceRoot = "source/engine";
    strictDeps = true;

    nativeBuildInputs = [
      cmake
      ninja
      cudaPackages.cuda_nvcc
      autoAddDriverRunpath
    ];

    buildInputs = [ cudaPackages.cudatoolkit zstd ];

    # upstream's cuda-compile release line (tools/packaging/build_cuda_linux.sh), one arch.
    # No GGML flags and no llama.cpp source: Strata vendors its ggml at
    # engine/third_party/strata-ggml (the commit engine/third_party/ggml/VERSION.txt names),
    # built by the engine's own CMake with its own defaults.
    cmakeFlags = [
      "-DCMAKE_GENERATOR=Ninja"
      "-DCMAKE_BUILD_TYPE=Release"
      "-DSTRATA_ENABLE_CUDA=ON"
      "-DSTRATA_BUILD_TESTS=OFF"
      "-DSTRATA_NATIVE_EXPERTS=ON"
      "-DSTRATA_PORTABLE=ON"
      "-DSTRATA_Q6K_EXPERTS=ON"
      "-DSTRATA_EXPERIMENTAL_SM60=OFF"
      # the engine components the families need, in one build tree: strata-hy4 needs
      # STRATA_BUILD_HY4 turned on (it links libzstd, hence zstd above; the bundled
      # third_party/zstd-dec is the fallback), strata-inkling builds by default.
      "-DSTRATA_BUILD_HY4=ON"
      "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
      "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
    ];

    env = {
      CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
      CUDAToolkit_INCLUDE_ROOT = cudaPackages.cudatoolkit;
      CUDAToolkit_LIBRARY_ROOT = cudaPackages.cudatoolkit;
    };

    # Named targets only, as upstream's own packaging scripts do. STRATA_BUILD_TESTS=OFF
    # does NOT gate the test programs: pinned_shared_test & co. are added unconditionally
    # on Linux (strata_gpu_engine.cmake) and fail to link against this build.
    # strata-q36 (Qwen3.6-35B-A3B) is built into the strata binary's own targets, not a
    # separate one: `strata` switches to the q36 path from the file's architecture.
    # strata-hy4 and strata-inkling are the engine binaries for their families.
    # Paths are absolute: the cmake env builder cd's into its build dir and stays there.
    buildPhase = ''
      runHook preBuild
      cmake --build "$NIX_BUILD_TOP/source/engine/build" \
        --target strata strata-device strata-hy4 strata-inkling -j "$NIX_BUILD_CORES"
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      # Do not `find ... -name strata`: the source tree ships shell launchers with that
      # name, and installing one over the engine breaks the service.
      cmake_dir="$NIX_BUILD_TOP/source/engine/build"
      # strata, strata-device and strata-hy4 link at the build dir's root; the inkling
      # subdirectory's CMake links strata-inkling into src/inkling2/.
      for path in strata strata-device strata-hy4 src/inkling2/strata-inkling; do
        if [[ ! -f "$cmake_dir/$path" ]]; then
          echo "compiled engine binary not found: $cmake_dir/$path" >&2
          exit 1
        fi
        install -Dm755 "$cmake_dir/$path" $out/bin/$(basename "$path")
      done

      cat > $out/bin/BUILD.json <<EOF
      {
        "version": "${version}",
        "archs": [${cudaArch}],
        "cuda": "${cudaPackages.cuda_nvcc.version}",
        "ptx": false,
        "source": "nix",
        "vision": "none",
        "ggml_pin": "${ggmlPin}"
      }
      EOF
      runHook postInstall
    '';

    passthru = {
      inherit cudaArch cudaVersion ggmlPin;
    };

    meta = {
      description = "Strata inference engine for Qwen3.8-Flash-Next (125B MoE) on a single NVIDIA GPU (sm_${cudaArch})";
      homepage = "https://github.com/Niko1221/Strata";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
      mainProgram = "strata";
    };
  }

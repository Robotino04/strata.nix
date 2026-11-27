{
  lib,
  cmake,
  ninja,
  patchelf,
  autoAddDriverRunpath,
  pkgs,
  src,
  version,
  cudaArch ? "89",
  cudaVersion ? "13",
}:
let
  # the same CUDA the engine is built with: a reader from a mismatched toolkit fails its
  # kernel loads on the card the engine runs on
  cudaPackages =
    if cudaVersion == "13"
    then pkgs.cudaPackages_13_0
    else if cudaVersion == "12"
    then pkgs.cudaPackages_12_9
    else throw "strata-vision: cudaVersion must be \"12\" (CUDA 12.9, sm_75..sm_89) or \"13\" (CUDA 13.0, sm_75..sm_120)";
in
  cudaPackages.backendStdenv.mkDerivation {
    pname = "strata-vision";
    inherit version;
    inherit src;
    # the vision tower is its own CMake project inside the source tree
    sourceRoot = "source/terminal/tools/vision";
    strictDeps = true;

    nativeBuildInputs = [
      cmake
      ninja
      patchelf
      cudaPackages.cuda_nvcc
      autoAddDriverRunpath
    ];

    buildInputs = [ cudaPackages.cudatoolkit ];

    # Strata's own Qwen vision tower (terminal/tools/vision): no llama.cpp, mtmd or ggml
    # is compiled into it (own-stack step 6), so the build is this one project alone.
    # LLAMA_DIR is read only with STRATA_VISION_ORACLES=ON, which stays off.
    # The CPU side keeps the portable AVX2 baseline (the CMakeLists' default, ON): the
    # binary runs on any x86-64 CPU and the tower picks its kernel tier at run time.
    # STRATA_PORTABLE=OFF (setup.py's native build) would bake the build machine's
    # -march into the binary, which has no place in the nix store.
    # fixupPhase matters here: without it the CMake CUDA glue leaves the toolkit's stub
    # `libcuda.so.1` in the binary's RUNPATH with nothing ahead of it, so ggml's
    # dlopen("libcuda.so.1") resolves the stub and CUDA init fails
    # ("CUDA driver is a stub library"). postFixup makes the real driver's path the
    # first RUNPATH entry for sure.
    phases = [ "unpackPhase" "configurePhase" "buildPhase" "installPhase" "fixupPhase" ];

    cmakeFlags = [
      "-DCMAKE_GENERATOR=Ninja"
      "-DCMAKE_BUILD_TYPE=Release"
      "-DSTRATA_VISION_CUDA=ON"
      "-DSTRATA_PORTABLE=ON"
      "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
      "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
    ];

    env = {
      CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
      CUDAToolkit_INCLUDE_ROOT = cudaPackages.cudatoolkit;
      CUDAToolkit_LIBRARY_ROOT = cudaPackages.cudatoolkit;
    };

    # the cmake env builder cd's into the sourceRoot and stays there, so the cmake
    # configure phase creates its build dir inside the sourceRoot (the same as the
    # engine's: $NIX_BUILD_TOP/source/engine/build). The vision CMake sets
    # RUNTIME_OUTPUT_DIRECTORY to ${CMAKE_BINARY_DIR}/bin, so the binary lands at
    # .../vision/build/bin/strata-vision.
    buildPhase = ''
      runHook preBuild
      ninja -j "$NIX_BUILD_CORES" -C "$NIX_BUILD_TOP/source/terminal/tools/vision/build" strata-vision
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      vision_dir="$NIX_BUILD_TOP/source/terminal/tools/vision/build"
      if [[ ! -f "$vision_dir/bin/strata-vision" ]]; then
        echo "compiled strata-vision binary not found: $vision_dir/bin/strata-vision" >&2
        exit 1
      fi
      install -Dm755 "$vision_dir/bin/strata-vision" $out/bin/strata-vision
      runHook postInstall
    '';

    postFixup = ''
      for f in $out/bin/*; do
        rpath=$(patchelf --print-rpath "$f" || true)
        case ":$rpath:" in
          *":/run/opengl-driver/lib:"*) ;;
          *) patchelf --set-rpath "/run/opengl-driver/lib''${rpath:+:$rpath}" "$f" ;;
        esac
      done
    '';

    passthru = {
      inherit cudaArch cudaVersion;
    };

    meta = {
      description = "Strata image encoder (Strata's own Qwen vision tower, CUDA sm_${cudaArch})";
      homepage = "https://github.com/Niko1221/Strata";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
      mainProgram = "strata-vision";
    };
  }

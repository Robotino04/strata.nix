{
  description = "Strata: run Qwen3.8-Flash-Next (125B MoE) locally on one NVIDIA GPU. NixOS package and service";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    strata = {
      url = "github:Niko1221/Strata/main";
      flake = false;
    };
    # Strata vendors everything it builds against (engine/third_party/strata-ggml,
    # strata-gguf-py, strata-llama): no llama.cpp source enters the build. This input
    # is the pin witness only — it must be the commit Strata's
    # engine/third_party/ggml/VERSION.txt names, so a vendor bump is visible here.
    llama-pinned = {
      url = "github:ggml-org/llama.cpp/3cf03257f219afbe7334045ff7c6a06ac68c627d";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    strata,
    llama-pinned,
  } @ inputs: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    version = let
      projectLine = builtins.head (builtins.filter
        (l: builtins.match "project\\(strata VERSION.*" l != null)
        (nixpkgs.lib.splitString "\n"
          (builtins.readFile (strata + "/engine/CMakeLists.txt"))));
    in
      builtins.head (builtins.match "project\\(strata VERSION ([0-9.]+).*" projectLine);

    ggmlPin =
      builtins.head (builtins.match "([0-9a-f]+).*"
        (builtins.readFile (strata + "/engine/third_party/ggml/VERSION.txt")));

    pinChecked =
      if (llama-pinned.rev or "") == ggmlPin
      then true
      else
        throw ''
          Strata's vendored ggml is frozen at ${ggmlPin}, but this flake locks llama-pinned at
          ${llama-pinned.rev or "an unknown revision"}.
          Fix: set inputs.llama-pinned.url = "github:ggml-org/llama.cpp/${ggmlPin}" in flake.nix,
          then run: nix flake lock --update-input llama-pinned'';

    strataPkgs = self.packages.${system};
  in {
    overlays.default = final: prev: {
      strata-engine = strataPkgs.strata-engine;
      strata-vision = strataPkgs.strata-vision;
      strata-serve = strataPkgs.strata-serve;
      strata = strataPkgs.strata;
    };

    packages.${system} = rec {
      strata-engine = pkgs.callPackage ./pkgs/engine.nix {
        inherit version ggmlPin;
        src = inputs.strata;
      };

      # Strata's own vision tower (terminal/tools/vision): no llama.cpp source.
      strata-vision = pkgs.callPackage ./pkgs/vision.nix {
        inherit version;
        src = inputs.strata;
      };

      strata-serve = pkgs.callPackage ./pkgs/serve.nix {
        inherit version pinChecked;
        strataSrc = inputs.strata;
        engine = strata-engine;
        vision = strata-vision;
      };

      strata = strata-serve;
      default = strata;
    };

    nixosModules.default = import ./nix/module.nix {
      strataPkgs = strataPkgs;
    };

    nixosModules.strata = self.nixosModules.default;

    nixosConfigurations.strata-eval = nixpkgs.lib.nixosSystem {
      system = system;
      modules = [
        self.nixosModules.default
        ({...}: {
          boot.isContainer = true;
          system.stateVersion = "26.11";
          services.strata.enable = true;
          services.strata.mockEngine = false;
          services.strata.prefetchOnStart = true;
        })
      ];
    };

    checks.${system} = {
      serve-smoke = pkgs.callPackage ./checks/serve-smoke.nix {
        strataPkg = strataPkgs.strata;
      };
      module-vm = pkgs.callPackage ./checks/vm.nix {
        strataModule = self.nixosModules.default;
      };
      module-vision = pkgs.callPackage ./checks/module-vision.nix {
        strataModule = self.nixosModules.default;
      };
      module-parallel = pkgs.callPackage ./checks/module-parallel.nix {
        strataModule = self.nixosModules.default;
      };
      module-extra-args = pkgs.callPackage ./checks/module-extra-args.nix {
        strataModule = self.nixosModules.default;
      };
      prefetch-vision = pkgs.callPackage ./checks/prefetch-vision.nix {
        strataPkg = strataPkgs.strata;
      };
    };

    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [ nixpkgs-fmt ];
    };
  };
}

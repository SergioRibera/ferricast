{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (baseSystem:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          system = baseSystem;
          inherit overlays;
          config.allowUnfree = true;
        };

        libraries = with pkgs; [
          # libappindicator
          #
          # gtk3
          # glib
          # cairo
          # gdk-pixbuf
          # libsoup_3
          # xdotool
          xorg.libX11
	  xorg.libXi
          xorg.libxcb
          libxkbcommon
	  x264.dev

          pipewire.dev
          pipewire
          libva
	  xorg.libXcursor

          wayland

          vulkan-loader
	  vulkan-validation-layers

          libGL
          fontconfig
          freetype
          pkgs.stdenv.cc.cc.lib

          dbus
          pipewire
          pkg-config

          # CUDA runtime stub. The real `libcuda.so.1` from the
          # proprietary NVIDIA driver lives at /run/opengl-driver/lib
          # on hosts with `hardware.nvidia.*` enabled and shadows
          # this stub at runtime via the LD_LIBRARY_PATH prepend
          # below. Listing the stub keeps the devshell usable on
          # boxes without the proprietary driver too.
          cudaPackages.cuda_cudart
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc

            cargo-dist
            cargo-release
            git-cliff
          ] ++ libraries;
          # `/run/opengl-driver/lib` is where NixOS hosts with
          # `hardware.opengl.enable` install GPU runtime libraries —
          # `libnvidia-encode.so.1` (NVENC) and `libcuda.so.1` among
          # them. It must come FIRST so the proprietary driver
          # shadows our nixpkgs CUDA stub.
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath libraries}";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  	  BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.glibc.dev}/include -I${pkgs.libclang.lib}/lib/clang/${pkgs.clang.version}/include";
        };
      });
}

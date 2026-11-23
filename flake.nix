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

        # One toolchain for the host build plus the standard libraries of
        # every platform the codec crates target, so `cargo check
        # --target …` verifies the Windows / Apple / Android backends from
        # Linux. Checking needs no SDK: the platform bindings (`windows`,
        # `objc2-*`, `ndk`) ship their declarations in Rust; only linking
        # needs the vendor toolchains. Pinned to the rustc nixpkgs shipped
        # before, so the switch to rust-overlay changes no lint or codegen.
        rustToolchain = pkgs.rust-bin.stable."1.95.0".default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
          targets = [
            "x86_64-pc-windows-msvc"
            "aarch64-pc-windows-msvc"
            "aarch64-apple-darwin"
            "x86_64-apple-darwin"
            "aarch64-apple-ios"
            "aarch64-apple-ios-sim"
            "aarch64-linux-android"
            "x86_64-linux-android"
          ];
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
          libx11
	  libxi
          libxcb
          libxkbcommon
	  x264.dev

          pipewire.dev
          pipewire
          libva

          # ALSA headers — rodio's audio output goes through cpal,
          # which on Linux talks to libasound. PipeWire installs an
          # alsa-emulation layer at runtime so playback still ends
          # up in the PipeWire graph, but the build-time dep is
          # alsa-lib regardless.
          alsa-lib
          alsa-lib.dev
	  libxcursor

          # Fraunhofer FDK AAC encoder. Used by the audio side of
          # the streaming pipeline (`crates/encoder/src/aac`) to
          # produce ADTS-framed AAC-LC for chromecast HLS. Required
          # to link `fdk-aac-sys` at build time; no runtime hook.
          fdk_aac

          wayland

          # gbm + libdrm: DMA-BUF allocation for the wayland-direct
          # capture backend. `libgbm` is mesa's general-purpose
          # buffer manager — present on every desktop Linux except
          # systems with *only* the proprietary NVIDIA driver
          # installed (NixOS hosts always have it because mesa is
          # used for software fallback / Xwayland regardless of GPU).
          # Newer nixpkgs split it out of `mesa` into its own
          # `libgbm` attribute; older ones still expose it via mesa.
          libgbm
          libdrm

          vulkan-loader
	  vulkan-validation-layers

          libGL
          fontconfig
          freetype
          pkgs.stdenv.cc.cc.lib

          dbus
          pipewire
          pkg-config

	  # used only for testing
	  #openh264
	  #gst_all_1.gstreamer
	  #gst_all_1.gst-plugins-good
	  #gst_all_1.gst-plugins-bad
	  #gst_all_1.gst-libav
	  #gst_all_1.gst-plugins-base

	  #gst_all_1.gst-plugins-ugly

	  


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
            rustToolchain

            cargo-dist
            cargo-release
            git-cliff

            # Freya shows a fatal panic in a dialog through rfd, which needs
            # zenity when the desktop portal cannot show message dialogs.
            zenity
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

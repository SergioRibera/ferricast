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

          dbus
          pipewire
          pkg-config
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
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath libraries}";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
	
  	  BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.glibc.dev}/include -I${pkgs.libclang.lib}/lib/clang/${pkgs.clang.version}/include";
        };
      });
}

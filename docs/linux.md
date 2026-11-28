# Linux

What a Linux user (x86-64, a Vulkan GPU) can do today, from one page: the build of the shaders and assets runs
natively, the Vulkan test tools build natively with GCC or Clang, and the loader (`gta6rt`) is a Windows program that
runs under Wine or Proton. Nothing is playable on any platform yet (README, *Project status*).

The package names below were checked on Ubuntu 26.04 LTS. Other distributions have the same tools under their own
names; the build lists anything it can't find before it starts.

## What runs where

| Piece | On Linux | How it is checked |
|---|---|---|
| `make_game.py`: check, shaders, assets, pipelines | Native | CI (`stdlib-tests-linux`, `translator-tests`, `semantic-tests`) |
| `Build.sh`, `Mods.sh` | Native (`/bin/sh`) | `tests/test_shell_launchers.py`, in the standard-library loop of `stdlib-tests-linux` |
| Shader translator, its tests and the GPU test tools (`vk_harness`, `vk_pipeline_check`, ...) | Native, GCC or Clang | CI (`linux-gpu-tools`: GCC, tests on llvmpipe); Clang builds them too, but no CI job uses it yet |
| `rpf9dump` (asset export) | Native, .NET 8 or newer | CI builds it on Windows only (the `check` job) |
| `gta6rt`, its shims and their tests | Windows program, under Wine or Proton | Cross-built with clang-cl and xwin; CI runs the Windows build under Wine on pushes to `main` |
| `scripts/run_game_wine.sh`: one retail run under Wine, headless or visible, counts only ([linux_proton.md](linux_proton.md#running-the-game-under-wine-with-one-command)) | Native (`bash`) around the Windows `gta6rt` | `tests/test_run_game_wine.py` and `tests/test_game_run_summary.py`, without the game |

## Build the port

Install the prerequisites (Ubuntu 26.04; the comment says what each is for):

```sh
sudo apt install git python3 \
  llvm-21 spirv-tools \
  dotnet-sdk-10.0 \
  cmake ninja-build g++ libvulkan-dev \
  ffmpeg zenity python3-tk
#   git, python3          the clone and make_game.py (Python 3.12 or newer)
#   llvm-21, spirv-tools  the shaders stage: llvm-mc with the AMDGPU target, spirv-as and spirv-val
#   dotnet-sdk-10.0       the assets stage builds rpf9dump; Ubuntu 26.04 packages no .NET 8 SDK, and 10 builds the
#                         net8.0 project through global.json's roll-forward
#   cmake ... libvulkan-dev  the pipelines stage builds vk_pipeline_check
#   ffmpeg                optional: XMA2 audio and --wav
#   zenity, python3-tk    optional: the --pick folder dialog and the mod manager window
```

Then, with your own copy of the game already readable on disk:

```sh
git clone --recursive https://github.com/ubiquitarianism/leonida
cd leonida
./Build.sh --find        # list the copies found: the home folder, Downloads, Documents, /mnt, /media/<user>, ...
./Build.sh --check       # check your copy, build nothing
./Build.sh               # build into ~/Documents/Leonida
```

`./Build.sh` is `python3 -I make_game.py` with the newest Python 3.12+ it finds, so every `make_game.py` option in
the README works with it (`--game`, `--out`, `--work`, `--stages`, ...). Without the cross-build tools below, the
default stages leave `runtime` out (`check,shaders,assets,pipelines`).

Notes:

- **llvm-mc**: Debian and Ubuntu's `llvm-21` installs it only as `llvm-mc-21` and `/usr/lib/llvm-21/bin/llvm-mc`.
  With `LLVM_MC` unset the tools use the plain `llvm-mc` when it is on `PATH` and otherwise the newest of those
  versioned names (`tools/mc_command.py`). Set `LLVM_MC` to choose another one.
- **File names are case-sensitive.** The build opens the package's files by their exact names
  (`MicrosoftGame.Config`, `Catalog000.bin`, ...). A copy whose names changed case on the way is reported with the
  name it found and the one it needs; nothing is renamed for you.
- **Mods**: `./Mods.sh` opens the mod manager window ([modding.md](modding.md)); `python3 -I tools/leonida_mods.py`
  is the same manager on the command line.

## The runtime, under Wine or Proton

`gta6rt` is a Windows program. To build it on Linux, add the LLVM cross tools and an MSVC CRT and Windows SDK laid
out by [xwin](https://github.com/Jake-Shadle/xwin) (running xwin downloads them from Microsoft and means accepting
Microsoft's licence):

```sh
sudo apt install clang-21 lld-21 llvm-21 wine
xwin --accept-license --sdk-version 10.0.26100 splat --use-winsysroot-style --preserve-ms-arch-notation \
     --output ~/.local/share/xwin-sdk
./Build.sh --stages runtime
```

On Ubuntu 24.04, `clang-cl-<N>` is in `clang-tools-<N>` instead ([testing.md](testing.md#known-problems)).
`make_game.py` finds the five tools under one version suffix (`clang-cl-21`, `lld-link-21`, `llvm-lib-21`,
`llvm-rc-21`, `llvm-mt-21`) or their plain names. Without the SDK it lists what is missing before any work starts.
`--runtime-build <folder>` takes a Windows build made elsewhere instead, such as the `runtime-windows` artifact of the
Runtime workflow. The stage writes `<out>/gta6rt.sh`, which starts `gta6rt.exe` with `wine` or the Wine named by `WINE`
(for Proton, its `files/bin/wine`); README, *Running the loader on Linux*, has the commands.
[testing.md](testing.md#the-windows-runtime-on-linux-under-wine-or-proton) has the test runs under Wine and Proton
and what fails there.

On Apple Silicon, set `LEONIDA_CX_BOTTLE` to the name of an existing CrossOver bottle to use the same launcher:

```sh
LEONIDA_CX_BOTTLE=Leonida ./gta6rt.sh --dry-run
```

Run this from the packaged output directory. The launcher uses CrossOver in `/Applications` by default. Set `WINE`
to the CrossOver `bin/wine` path if the app is installed elsewhere. It enables Rosetta's AVX advertisement by default
for this mode and honours an explicit `ROSETTA_ADVERTISE_AVX` value.

For XMA2 audio, `runtime/CMakeLists.txt` builds `runtime/xma_decoder` for `x86_64-pc-windows-msvc` when `cargo` is on
`PATH`. Distribution Rust packages typically omit the Windows MSVC standard library, which leaves XMA voices silent
unless you add the `x86_64-pc-windows-msvc` Rust target or pass a prebuilt static library with
`-DLEONIDA_XMA2_DECODER_LIB=<path>`, built yourself from this checkout's `runtime/xma_decoder` with the command in
[runtime/xma_decoder/README.md](../runtime/xma_decoder/README.md).

### The D3D12 runtime (`leonida.exe`) under Wine

`scripts/play-linux.sh` and PLAY in the launcher start `leonida.exe`, the D3D12 runtime in `d3d12/`. `./Build.sh`
does not build it. On Linux:

```sh
cargo install --locked cargo-xwin                       # once; also needs clang and lld
XWIN_ACCEPT_LICENSE=1 sh scripts/build-d3d12-linux.sh   # leonida.exe, VKD3D-Proton and DXVK's dxgi.dll
sh scripts/play-linux.sh /path/to/game_scarlett_master.exe
```

cargo-xwin downloads Microsoft's CRT and Windows SDK on its first run. `XWIN_ACCEPT_LICENSE=1` accepts their licence;
without it the script stops before downloading. The runtime lands in `d3d12/target/x86_64-pc-windows-msvc/diag/`,
VKD3D-Proton in `d3d12/target/vkd3d-proton/` (fetching it needs Python 3.14 or newer), DXVK's `dxgi.dll` in
`d3d12/target/dxvk/` (`scripts/fetch_dxvk.py`, which fetches it on its own too). `play-linux.sh` looks for
`leonida.exe` beside itself, then under `d3d12/target/`. `LEONIDA_RUNTIME` and `LEONIDA_VKD3D` select other files.

The script takes `LEONIDA_PROTON` (a Proton folder or its `files/bin/wine`), else `LEONIDA_WINE` (a Wine program, such
as `wine`), else the usable Proton build with the highest version, else `wine` on `PATH`. Every folder with
`files/bin/wine` counts as a Proton build, whatever its name, in Steam's `compatibilitytools.d` and `steamapps/common`
(all libraries, Flatpak and Snap too), `STEAM_EXTRA_COMPAT_TOOLS_PATHS` and the system's `compatibilitytools.d`; it is
usable when it also has `files/lib/wine` or `files/lib64/wine`; others are left out. Builds are ranked by the numbers
in their version. `--find-proton` prints the Proton or Wine the script would use, `--list-protons` every usable build;
neither changes anything.

The game needs DXVK's `dxgi.dll`: with Wine's own (on `wined3d`) it crashes when it creates its swap chain, and Wine's
own `d3d12.dll` (vkd3d) creates no device for it. The `dxgi.dll` comes from the Proton; if it has none, or under the
system's Wine, from `LEONIDA_DXVK` (a folder with a 64-bit `dxgi.dll`), else the package (`dxvk/` beside the script,
then `d3d12/target/dxvk/`), else the distribution's DXVK (under `/usr/share/dxvk` or `/usr/lib/dxvk`), else another
Proton. Without any, the script stops and names `fetch_dxvk.py`. It copies that dxgi.dll and the bundled VKD3D-Proton
into its own prefix (`LEONIDA_WINEPREFIX`, made and updated with the Wine in use) and sets
`WINEDLLOVERRIDES=d3d12,d3d12core,dxgi=n` ([boot.md](boot.md#under-wine-on-linux)). Under a Proton it puts the build's
libraries first in `WINEDLLPATH` and `LD_LIBRARY_PATH`, as Proton's own `proton` script does. VKD3D-Proton's shader
cache goes to `<LEONIDA_CACHE_DIR>/vkd3d-proton`.

VKD3D-Proton releases without dxil-spirv's new structurizer turn some of the title's compute shaders into invalid
SPIR-V, and the Vulkan driver crashes on them (`vkCreateComputePipelines`, exception `0xc0000005`). When the bundled
`d3d12core.dll` lacks it, `play-linux.sh` takes the VKD3D-Proton of the Proton in use, else of another Proton found that
has it (highest version first), also when `LEONIDA_PROTON` chose the Proton. If none has it, the script warns and uses
the bundled one. `LEONIDA_VKD3D` selects a VKD3D-Proton and skips this check.

### Why there is no native Linux `gta6rt`

A native (ELF) loader is not a matter of build flags:

- The loader shims only the console's own modules and a few Win32 entry points it has to own (GDK, XMem,
  `d3d12_xs`, the library loader, the process and file-system shims). Every other import binds to the host's Windows
  DLLs unchanged (`runtime/src/pe_loader.cpp`; [runtime_imports.md](runtime_imports.md): `XboxBvhBuilder.dll`
  alone binds 4 imports to shims and 53 to the host). On Linux, those host DLLs are what Wine provides.
- The mapped code uses the Windows x64 calling convention and the Windows thread and exception model: the loader
  registers unwind tables with `RtlAddFunctionTable`, keeps a module TLS slot through `gta6rt_tls_host.dll`, and
  `main.cpp` installs a vectored exception handler. The shims are written against `windows.h`.

A Linux-native host would have to reimplement each of those Win32 APIs, which is the work Wine already does, so the
supported way is the Windows build under Wine or Proton with the D3D12 layer on native Vulkan. What does build
natively is everything that doesn't touch Win32: the Vulkan tools and the portable runtime pieces with their tests
(`ctest` on a Linux build runs them, including the root-signature model and the host CPU check).

## Build and test without the game

```sh
cmake -S runtime -B runtime/build/linux -G Ninja                 # CC=clang CXX=clang++ for Clang
cmake --build runtime/build/linux --parallel
ctest --test-dir runtime/build/linux --output-on-failure         # Vulkan tests run on Mesa's lavapipe without a GPU
python3 -I scripts/run_semantic_tests.py tests/semantic          # translator suite
python3 -I tests/test_<name>.py                                  # one standard-library test
```

[testing.md](testing.md) covers the GPU tests (`scripts/run_gpu_tests.py`), running them on llvmpipe, and choosing
the GPU.

### Without root (micromamba)

If you cannot install distribution packages (no `sudo`), you can install `llvm-tools` and `spirv-tools` into a
user-owned folder with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
(x86-64 glibc; for ARM64 use `linux-aarch64` from the micromamba guide; musl/Alpine is not natively supported) and let
CMake fetch its pinned, hash-checked Vulkan headers if the system does not provide them:

```bash
LEO=$HOME/leonida-tools
mkdir -p "$LEO/bin" "$LEO/mamba"
# Note: `latest` is unpinned and downloaded without signature verification; inspect or pin per the micromamba guide.
curl -fsSLo "$LEO/micromamba.tar.bz2" https://micro.mamba.pm/api/micromamba/linux-64/latest
tar -xjf "$LEO/micromamba.tar.bz2" -C "$LEO/bin" --strip-components=1 bin/micromamba

export MAMBA_ROOT_PREFIX="$LEO/mamba"
"$LEO/bin/micromamba" create -y -n leonida -c conda-forge --override-channels "llvm-tools>=21" spirv-tools
export PATH="$LEO/mamba/envs/leonida/bin:$PATH"

cmake -S runtime -B runtime/build/linux -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLEONIDA_VULKAN_HEADERS_FETCH=ON
cmake --build runtime/build/linux --parallel "$(nproc)"
LLVM_MC=llvm-mc LEONIDA_VK_DEVICE="<part of your GPU name>" \
  python3 -I scripts/run_gpu_tests.py --build runtime/build/linux --work "$LEO/run" --require-pass | tee "$LEO/run.log"
echo "runner exit: ${PIPESTATUS[0]}"
```

Notes:
- `LEONIDA_VK_DEVICE` selects the physical GPU by substring so the runner does not fall back to Mesa's `llvmpipe`.
- Keep `$LEO` on a filesystem without `noexec` (`findmnt -no OPTIONS "$LEO"`).
- System prerequisites still needed: a working Vulkan driver/loader (`vulkaninfo --summary`), `cmake`, `ninja`, `g++`
  (or `clang++`), `python3`, `curl`, `tar`, and `bzip2`.
- Cleanup: delete `$LEO` and `runtime/build/linux`. micromamba also writes `~/.conda/environments.txt` and
  `~/.mamba/pkgs/urls.txt` (delete only those files if you already use conda).

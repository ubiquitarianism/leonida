#!/bin/sh
set -eu

here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
root=$(dirname -- "$here")
d3d12=$root/d3d12
target=x86_64-pc-windows-msvc
features=diagnostics,mimalloc
vkd3d=yes
for arg in "$@"; do
    case $arg in
        --no-vkd3d) vkd3d=no ;;
        *) echo "usage: $0 [--no-vkd3d]" >&2; exit 2 ;;
    esac
done

missing=
command -v cargo >/dev/null 2>&1 || missing="$missing rustup/cargo (https://rustup.rs)"
command -v cargo-xwin >/dev/null 2>&1 || missing="$missing cargo-xwin (cargo install --locked cargo-xwin)"
command -v clang >/dev/null 2>&1 || missing="$missing clang"
command -v lld-link >/dev/null 2>&1 || command -v ld.lld >/dev/null 2>&1 || missing="$missing lld"
if [ -n "$missing" ]; then
    echo "Building leonida.exe on Linux needs:$missing" >&2
    exit 1
fi
if [ "${XWIN_ACCEPT_LICENSE:-}" != 1 ]; then
    echo "cargo-xwin downloads Microsoft's CRT and Windows SDK (once, into its cache). Read" >&2
    echo "https://go.microsoft.com/fwlink/?LinkId=2086102 and, if you accept it, run again with" >&2
    echo "XWIN_ACCEPT_LICENSE=1 sh $0" >&2
    exit 1
fi

echo "Building leonida.exe ($target, profile diag, features $features)"
(cd "$d3d12" && rustup target add "$target" >/dev/null 2>&1 || true
 cd "$d3d12" && cargo xwin build --locked --target "$target" --profile diag -p leonida --features "$features")
exe=$d3d12/target/$target/diag/leonida.exe
[ -f "$exe" ] || { echo "cargo finished without $exe" >&2; exit 1; }
printf '%s\n' "$features" > "$d3d12/target/$target/diag/built-with.txt"
echo "Built: $exe"

if [ "$vkd3d" = yes ]; then
    out=$d3d12/target/vkd3d-proton
    if [ -f "$out/d3d12.dll" ] && [ -f "$out/d3d12core.dll" ]; then
        echo "VKD3D-Proton: already in $out"
    else
        py=
        for candidate in python3.16 python3.15 python3.14 python3; do
            command -v "$candidate" >/dev/null 2>&1 || continue
            if "$candidate" -c 'import compression.zstd' >/dev/null 2>&1; then py=$candidate; break; fi
        done
        if [ -z "$py" ]; then
            echo "VKD3D-Proton not fetched: scripts/fetch_vkd3d_proton.py needs Python 3.14 or newer." >&2
            echo "Install it and run" >&2
            echo "  python3.14 -I $here/fetch_vkd3d_proton.py --out $out" >&2
            exit 1
        fi
        "$py" -I "$here/fetch_vkd3d_proton.py" --out "$out"
    fi
    out=$d3d12/target/dxvk
    if [ -f "$out/dxgi.dll" ]; then
        echo "DXVK: already in $out"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -I "$here/fetch_dxvk.py" --out "$out"
    else
        echo "DXVK not fetched (needed under the system's Wine, not under Proton): run" >&2
        echo "  python3 -I $here/fetch_dxvk.py --out $out" >&2
    fi
fi
echo "Next: sh $here/play-linux.sh /path/to/game_scarlett_master.exe (or PLAY in the launcher)"

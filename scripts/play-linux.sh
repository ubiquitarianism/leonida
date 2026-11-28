#!/bin/sh
# Starts Leonida's D3D12 runtime (d3d12/, a Windows program) on Linux under an installed Proton build or the system's
# Wine, with the VKD3D-Proton that comes in the package (scripts/fetch_vkd3d_proton.py puts it in vkd3d-proton/ beside
# this script) and DXVK's dxgi.dll (docs/linux.md says where both come from). A Vulkan driver (Mesa or NVIDIA's) is
# the system's.
#
#   ./play-linux.sh /path/to/your/game_scarlett_master.exe [--series-x] [--log]
#   ./play-linux.sh --find-proton | --list-protons
#
# Leonida keeps its own Wine prefix (LEONIDA_WINEPREFIX, default ~/.local/share/leonida/prefix): the bundled
# d3d12.dll and d3d12core.dll are installed there and preferred over Wine's (WINEDLLOVERRIDES), so no other prefix
# is changed. LEONIDA_* variables already set win over the defaults below.
set -eu

proton_problem() {
    if [ ! -x "$1/files/bin/wine" ]; then
        echo "no files/bin/wine"
    elif [ ! -d "$1/files/lib/wine" ] && [ ! -d "$1/files/lib64/wine" ]; then
        echo "incomplete, no files/lib/wine or files/lib64/wine"
    fi
}

proton_ok() {
    [ -z "$(proton_problem "$1")" ]
}

proton_dxvk() {
    for lib_dir in "$1/files/lib/wine/dxvk/x86_64-windows" "$1/files/lib64/wine/dxvk/x86_64-windows" \
                   "$1/files/lib64/wine/dxvk"; do
        if [ -f "$lib_dir/dxgi.dll" ]; then
            printf '%s\n' "$lib_dir"
            return 0
        fi
    done
    return 1
}

proton_new_vkd3d() {
    for lib_dir in "$1/files/lib/wine/vkd3d-proton/x86_64-windows" "$1/files/lib64/wine/vkd3d-proton/x86_64-windows" \
                   "$1/files/lib64/wine/vkd3d-proton"; do
        if vkd3d_new_structurizer "$lib_dir"; then
            printf '%s\n' "$lib_dir"
            return 0
        fi
    done
    return 1
}

proton_name() {
    name=
    if [ -f "$1/version" ]; then
        name=$(awk 'NR == 1 { print (NF > 1 ? $2 : $1) }' "$1/version")
    fi
    [ -n "$name" ] || name=$(basename "$1")
    printf '%s\n' "$name"
}

steam_roots() {
    for root in "${XDG_DATA_HOME:-$HOME/.local/share}/Steam" "$HOME/.steam/root" "$HOME/.steam/steam" \
                "$HOME/.steam/debian-installation" "$HOME/.var/app/com.valvesoftware.Steam/data/Steam" \
                "$HOME/snap/steam/common/.local/share/Steam"; do
        if [ -d "$root" ]; then
            printf '%s\n' "$root"
            for vdf in "$root/steamapps/libraryfolders.vdf" "$root/config/libraryfolders.vdf"; do
                if [ -f "$vdf" ]; then
                    sed -n 's/^[[:space:]]*"path"[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "$vdf"
                fi
            done
        fi
    done
}

proton_candidates() {
    {
        steam_roots | while IFS= read -r root; do
            for dir in "$root"/compatibilitytools.d/* "$root"/steamapps/common/*; do
                printf '%s\n' "$dir"
            done
        done
        printf '%s\n' "${STEAM_EXTRA_COMPAT_TOOLS_PATHS:-}" | tr ':' '\n' | while IFS= read -r tools; do
            if [ -n "$tools" ]; then
                printf '%s\n' "$tools"
                for dir in "$tools"/*; do
                    printf '%s\n' "$dir"
                done
            fi
        done
        for tools in /usr/share/steam/compatibilitytools.d /usr/local/share/steam/compatibilitytools.d; do
            for dir in "$tools"/*; do
                printf '%s\n' "$dir"
            done
        done
    } | while IFS= read -r dir; do
        if [ -d "$dir" ] && [ -e "$dir/files/bin/wine" ]; then
            real=$(cd -P "$dir" && pwd)
            printf '%s\t%s\t%s\n' "$(proton_name "$real")" "$real" "$(proton_problem "$real")"
        fi
    done
}

all_protons() {
    proton_candidates | awk -F '\t' '{
        key = $1
        gsub(/[^0-9]+/, " ", key)
        split(key, part, " ")
        printf "%d %012d %012d %012d\t%s\t%s%s\n", ($3 == ""), part[1], part[2], part[3], $1, $2,
               ($3 == "" ? "" : "\t" $3)
    }' | sort -r -u | cut -f 2-
}

ranked_protons() {
    all_protons | awk -F '\t' '$3 == "" { print $1 "\t" $2 }'
}

newest_proton() {
    ranked_protons | head -n 1
}

vkd3d_new_structurizer() {
    [ -f "$1/d3d12.dll" ] && [ -f "$1/d3d12core.dll" ] && grep -q -a legacy_structurizer "$1/d3d12core.dll"
}

proton_with_new_vkd3d() {
    { [ -z "$proton" ] || printf '%s\n' "$proton"; ranked_protons | cut -f 2-; } |
    while IFS= read -r dir; do
        if proton_new_vkd3d "$dir" >/dev/null; then
            printf '%s\n' "$dir"
            break
        fi
    done
}

system_wine() {
    wine=$(command -v "$1" 2>/dev/null) || return 1
    [ -f "$wine" ] && [ -x "$wine" ] || return 1
    version=$("$wine" --version 2>/dev/null | head -n 1) || version=
    wine_label="Wine${version:+ ${version#wine-}}"
}

system_dxvk() {
    for dir in /usr/share/dxvk/x64 /usr/lib/dxvk/x64 /usr/lib/dxvk/wine64 /usr/share/dxvk/x86_64-windows \
               /usr/lib/dxvk/x86_64-windows; do
        if [ -f "$dir/dxgi.dll" ]; then
            printf '%s\n' "$dir"
            return
        fi
    done
}

find_dxgi() {
    if [ -n "${LEONIDA_DXVK:-}" ]; then
        dxgi_dir=${LEONIDA_DXVK%/}
        if [ ! -f "$dxgi_dir/dxgi.dll" ]; then
            echo "Leonida: LEONIDA_DXVK=$LEONIDA_DXVK has no dxgi.dll" >&2
            exit 1
        fi
        dxgi_label="DXVK's dxgi from $dxgi_dir (LEONIDA_DXVK)"
        return
    fi
    if [ -n "$proton" ] && dxgi_dir=$(proton_dxvk "$proton"); then
        dxgi_label="DXVK's dxgi"
        return
    fi
    for dxgi_dir in "$here/dxvk" "$root/d3d12/target/dxvk"; do
        if [ -f "$dxgi_dir/dxgi.dll" ]; then
            dxgi_label="the bundled DXVK's dxgi ($(sed -n '1s/,.*//p' "$dxgi_dir/SOURCE.txt" 2>/dev/null | grep . || echo bundled))"
            return
        fi
    done
    dxgi_dir=$(system_dxvk)
    if [ -n "$dxgi_dir" ]; then
        dxgi_label="DXVK's dxgi from $dxgi_dir"
        return
    fi
    found=$(all_protons | cut -f 1,2 | while IFS= read -r line; do
        if proton_dxvk "$(printf '%s\n' "$line" | cut -f 2-)" >/dev/null; then printf '%s\n' "$line"; break; fi
    done)
    if [ -n "$found" ]; then
        dxgi_dir=$(proton_dxvk "$(printf '%s\n' "$found" | cut -f 2-)")
        dxgi_label="DXVK's dxgi of $(printf '%s\n' "$found" | cut -f 1)"
        return
    fi
    cat >&2 <<MESSAGE
Leonida: $wine_label ($wine) needs DXVK's dxgi.dll: with Wine's own dxgi.dll the game crashes when it creates its
swap chain. Fetch it with
  python3 -I $here/fetch_dxvk.py --out $root/d3d12/target/dxvk
or install your distribution's DXVK or a Proton build, or set LEONIDA_DXVK to a folder with DXVK's 64-bit dxgi.dll.
MESSAGE
    exit 1
}

find_proton() {
    proton=
    if [ -n "${LEONIDA_PROTON:-}" ]; then
        proton=${LEONIDA_PROTON%/}
        case $proton in */files/bin/wine) proton=${proton%/files/bin/wine} ;; esac
        if [ ! -d "$proton" ]; then
            echo "Leonida: LEONIDA_PROTON=$LEONIDA_PROTON is not a Proton build (no such folder)" >&2
            exit 1
        fi
        problem=$(proton_problem "$proton")
        if [ -n "$problem" ]; then
            echo "Leonida: LEONIDA_PROTON=$LEONIDA_PROTON is not a Proton build ($problem)" >&2
            exit 1
        fi
        proton=$(cd -P "$proton" && pwd)
        wine_label=$(proton_name "$proton")
    elif [ -n "${LEONIDA_WINE:-}" ]; then
        if ! system_wine "$LEONIDA_WINE"; then
            echo "Leonida: LEONIDA_WINE=$LEONIDA_WINE is not a Wine program" >&2
            exit 1
        fi
    else
        found=$(newest_proton)
        if [ -n "$found" ]; then
            wine_label=$(printf '%s\n' "$found" | cut -f 1)
            proton=$(printf '%s\n' "$found" | cut -f 2-)
        elif ! system_wine wine; then
            cat >&2 <<'MESSAGE'
Leonida: no Proton and no Wine found. Install GE-Proton, Valve's Proton or your distribution's Wine, or set
LEONIDA_PROTON to a Proton folder or LEONIDA_WINE to a Wine program.
MESSAGE
            exit 1
        fi
    fi
    if [ -n "$proton" ]; then
        wine=$proton/files/bin/wine
        runner=$proton
    else
        runner=$wine
    fi
    find_dxgi
}

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")

if [ "${1:-}" = "--list-protons" ]; then
    ranked_protons
    exit 0
fi

if [ "${1:-}" = "--find-proton" ]; then
    find_proton
    printf '%s\t%s\n' "$wine_label" "$runner"
    exit 0
fi

if [ $# -lt 1 ]; then
    echo "usage: $0 /path/to/game_scarlett_master.exe [--series-x] [--log]" >&2
    exit 2
fi
game_exe=$1
shift
console=series-s
log=
for arg in "$@"; do
    case $arg in
        --series-x) console=series-x ;;
        --log) log=1 ;;
        *) echo "usage: $0 /path/to/game_scarlett_master.exe [--series-x] [--log]" >&2; exit 2 ;;
    esac
done
[ -f "$game_exe" ] || { echo "No $game_exe" >&2; exit 1; }

build="sh $here/build-d3d12-linux.sh"
if [ -n "${LEONIDA_RUNTIME:-}" ]; then
    runtime=$LEONIDA_RUNTIME
    [ -f "$runtime" ] || { echo "No $runtime (LEONIDA_RUNTIME)" >&2; exit 1; }
else
    runtime=
    for candidate in "$here/leonida.exe" \
        "$root/d3d12/target/x86_64-pc-windows-msvc/diag/leonida.exe" \
        "$root/d3d12/target/x86_64-pc-windows-msvc/play/leonida.exe" \
        "$root/d3d12/target/x86_64-pc-windows-msvc/release/leonida.exe" \
        "$root/d3d12/target/diag/leonida.exe" \
        "$root/d3d12/target/play/leonida.exe" \
        "$root/d3d12/target/release/leonida.exe"; do
        if [ -f "$candidate" ]; then runtime=$candidate; break; fi
    done
    if [ -z "$runtime" ]; then
        echo "No leonida.exe beside this script or in $root/d3d12/target/. Build it with" >&2
        echo "  XWIN_ACCEPT_LICENSE=1 $build" >&2
        echo "(in the launcher: Tools, Rebuild the runtime), or set LEONIDA_RUNTIME to a leonida.exe." >&2
        exit 1
    fi
fi
if [ -n "${LEONIDA_VKD3D:-}" ]; then
    vkd3d=$LEONIDA_VKD3D
else
    vkd3d=$here/vkd3d-proton
    if [ ! -f "$vkd3d/d3d12.dll" ] && [ -f "$root/d3d12/target/vkd3d-proton/d3d12.dll" ]; then
        vkd3d=$root/d3d12/target/vkd3d-proton
    fi
fi
find_proton
if [ -n "${LEONIDA_VKD3D:-}" ]; then
    vkd3d_label="the VKD3D-Proton in $vkd3d (LEONIDA_VKD3D)"
else
    vkd3d_label="the bundled VKD3D-Proton ($(sed -n '1s/,.*//p' "$vkd3d/SOURCE.txt" 2>/dev/null | grep . || echo bundled))"
fi
if [ -z "${LEONIDA_VKD3D:-}" ] && ! vkd3d_new_structurizer "$vkd3d"; then
    vkd3d_from=$(proton_with_new_vkd3d)
    if [ -n "$vkd3d_from" ]; then
        vkd3d=$(proton_new_vkd3d "$vkd3d_from")
        vkd3d_build=$(sed -n '1s/.*(\(.*\)).*/\1/p' "${vkd3d%/x86_64-windows}/version" 2>/dev/null || true)
        vkd3d_label="the VKD3D-Proton of $(proton_name "$vkd3d_from")${vkd3d_build:+ ($vkd3d_build)}"
    elif [ -f "$vkd3d/d3d12core.dll" ]; then
        echo "Leonida: the bundled VKD3D-Proton lacks dxil-spirv's new structurizer, and the Vulkan driver can" \
             "crash on some of the title's compute shaders. No Proton build found has a newer one. Install a" \
             "current GE-Proton, or set LEONIDA_VKD3D to a newer VKD3D-Proton folder." >&2
    fi
fi
for dll in d3d12.dll d3d12core.dll; do
    if [ ! -f "$vkd3d/$dll" ]; then
        echo "No $vkd3d/$dll (VKD3D-Proton). Fetch it with" >&2
        echo "  XWIN_ACCEPT_LICENSE=1 $build" >&2
        echo "or python3.14 -I $here/fetch_vkd3d_proton.py --out $root/d3d12/target/vkd3d-proton," >&2
        echo "or set LEONIDA_VKD3D to a folder with d3d12.dll and d3d12core.dll." >&2
        exit 1
    fi
done
if [ -n "$proton" ]; then
    for dir in "$proton/files/lib/wine" "$proton/files/lib/vkd3d" "$proton/files/lib64/wine" \
               "$proton/files/lib64/vkd3d"; do
        if [ -d "$dir" ]; then WINEDLLPATH=$dir${WINEDLLPATH:+:$WINEDLLPATH}; fi
    done
    for dir in "$proton/files/lib/i386-linux-gnu" "$proton/files/lib/x86_64-linux-gnu" "$proton/files/lib64"; do
        if [ -d "$dir" ]; then LD_LIBRARY_PATH=$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
    done
    export WINEDLLPATH LD_LIBRARY_PATH
fi

data=${XDG_DATA_HOME:-$HOME/.local/share}/leonida
export WINEPREFIX=${LEONIDA_WINEPREFIX:-$data/prefix}
export WINEARCH=win64
stamp=$WINEPREFIX/.leonida-proton
system32=$WINEPREFIX/drive_c/windows/system32
if [ ! -d "$system32" ] || [ "$(cat "$stamp" 2>/dev/null)" != "$wine_label $runner" ] ||
   { [ -n "$proton" ] && { [ -f "$proton/files/lib/vkd3d/x86_64-windows/libvkd3d-1.dll" ] ||
                           [ -f "$proton/files/lib64/vkd3d/x86_64-windows/libvkd3d-1.dll" ]; } &&
     [ ! -f "$system32/libvkd3d-1.dll" ]; }; then
    if [ -d "$system32" ]; then
        echo "Leonida: updating its Wine prefix in $WINEPREFIX for $wine_label"
    else
        echo "Leonida: making its Wine prefix in $WINEPREFIX with $wine_label"
        mkdir -p "$WINEPREFIX"
    fi
    WINEDEBUG=-all "$wine" wineboot -u >/dev/null 2>&1 || {
        echo "Leonida: $wine_label could not set up the prefix $WINEPREFIX" >&2
        exit 1
    }
    server=$(dirname "$wine")/wineserver
    if [ ! -x "$server" ] && [ -z "$proton" ]; then server=$(command -v wineserver 2>/dev/null || true); fi
    if [ -n "$server" ] && [ -x "$server" ]; then
        "$server" -w || true
    fi
    printf '%s\n' "$wine_label $runner" > "$stamp"
fi
# The bundled VKD3D-Proton, copied in when it is missing or differs (a new package).
for dll in d3d12.dll d3d12core.dll; do
    if ! cmp -s "$vkd3d/$dll" "$system32/$dll"; then cp "$vkd3d/$dll" "$system32/$dll"; fi
done
if ! cmp -s "$dxgi_dir/dxgi.dll" "$system32/dxgi.dll"; then cp "$dxgi_dir/dxgi.dll" "$system32/dxgi.dll"; fi
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"

# Where the title's files are, and where the caches and saves go.
export LEONIDA_GAME_DIR=${LEONIDA_GAME_DIR:-$(cd "$(dirname "$game_exe")" && pwd)}
export LEONIDA_CACHE_DIR=${LEONIDA_CACHE_DIR:-$data/cache}
export LEONIDA_DATA_DIR=${LEONIDA_DATA_DIR:-$data/data}
export VKD3D_SHADER_CACHE_PATH=${VKD3D_SHADER_CACHE_PATH:-$LEONIDA_CACHE_DIR/vkd3d-proton}
mkdir -p "$LEONIDA_CACHE_DIR" "$LEONIDA_DATA_DIR" "$VKD3D_SHADER_CACHE_PATH"

# The settings scripts/play-nvidia.ps1 measured (d3d12/README.md), as defaults.
set_default() { eval "[ -n \"\${$1:-}\" ] || export $1=\"$2\""; }
set_default LEONIDA_TEXTURE_NO_WAIT 1
set_default LEONIDA_RELAY 2
set_default LEONIDA_CARD_STEPS 2
set_default LEONIDA_DYNAMIC_SLOTS 524288
set_default LEONIDA_PIPELINE_WAIT_MS 10000
set_default LEONIDA_RENDER_SCALE 0.5
set_default LEONIDA_OWN_UPSCALER 1
set_default LEONIDA_BVH_MISS 1
set_default LEONIDA_CARD_BUFFERS_MIN_MB 1
set_default LEONIDA_CARD_BUFFERS_MB 1536
set_default LEONIDA_COMPOSE 1
set_default LEONIDA_NAN_SCRUB 1
set_default LEONIDA_AMBIENT_AIR 1.5
set_default LEONIDA_AMBIENT_MOST 2

echo "Leonida: $console under $wine_label ($runner) with $dxgi_label and $vkd3d_label"
if [ -n "$log" ]; then
    logs=$data/logs
    mkdir -p "$logs"
    ls -1t "$logs"/play-*.log 2>/dev/null | tail -n +5 | while IFS= read -r old; do rm -f "$old"; done
    log_file=$logs/play-$(date +%m%d-%H%M%S).log
    echo "log: $log_file"
    exec "$wine" "$runtime" run "$game_exe" --execute --console "$console" 2>"$log_file"
fi
exec "$wine" "$runtime" run "$game_exe" --execute --console "$console"

#!/usr/bin/env python3
import argparse
import hashlib
import io
import os
import sys
import tarfile
import urllib.request

VERSION = "3.1.1"
RELEASE = (f"https://github.com/doitsujin/dxvk/releases/download/v{VERSION}/dxvk-{VERSION}.tar.gz",
           "40565b4a724aadc4433fa4e010b4b23916d9b1f1baeee64e17186db94f54e608")
LICENSE = (f"https://raw.githubusercontent.com/doitsujin/dxvk/v{VERSION}/LICENSE",
           "a5cb1a6ded7d2d7e92d550ba28edd21be2d1d4044662b399887351023e30ce64")
DLLS = ("dxgi.dll",)


def fetch(url, sha256):
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise SystemExit(f"{url}: SHA-256 {got}, expected {sha256}; not used")
    return data


def main():
    parser = argparse.ArgumentParser(description="Fetch the pinned DXVK release for Leonida's Linux package.")
    parser.add_argument("--out", default=os.path.join("dist", "linux", "dxvk"),
                        help="folder for the DLL, license and source note (created)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(fetch(*RELEASE)), mode="r:gz") as tar:
        for name in DLLS:
            member = tar.getmember(f"dxvk-{VERSION}/x64/{name}")
            with tar.extractfile(member) as src, open(os.path.join(args.out, name), "wb") as dst:
                dst.write(src.read())
    with open(os.path.join(args.out, "LICENSE"), "wb") as dst:
        dst.write(fetch(*LICENSE))
    with open(os.path.join(args.out, "SOURCE.txt"), "w", encoding="utf-8", newline="\n") as note:
        note.write(f"DXVK {VERSION}, unmodified 64-bit {', '.join(DLLS)} from\n"
                   f"{RELEASE[0]}\n(SHA-256 {RELEASE[1]}).\n"
                   f"Licensed under the zlib/libpng license (LICENSE).\n"
                   f"Source: https://github.com/doitsujin/dxvk/tree/v{VERSION}\n")
    print(f"DXVK {VERSION}: {', '.join(DLLS)}, LICENSE and SOURCE.txt in {args.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())

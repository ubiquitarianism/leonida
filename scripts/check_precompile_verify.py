"""Exercise gta6rt's offline journal writer and verifier in separate processes using two authored compute shaders."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from spirv_asm import assemble


SHADER = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""

# Write a literal through a device address in the push constants. Unlike the empty shader, this has an
# observable side effect. Both use the runtime's plain compute layout and are never dispatched here.
STORE_SHADER = """OpCapability Shader
OpCapability PhysicalStorageBufferAddresses
OpExtension "SPV_KHR_physical_storage_buffer"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %main "main" %push
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %args Block
OpMemberDecorate %args 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%address = OpTypeVector %uint 2
%args = OpTypeStruct %address
%p_args = OpTypePointer PushConstant %args
%p_address = OpTypePointer PushConstant %address
%p_word = OpTypePointer PhysicalStorageBuffer %uint
%zero = OpConstant %uint 0
%value = OpConstant %uint {value}
%push = OpVariable %p_args PushConstant
%main = OpFunction %void None %fn
%entry = OpLabel
%field = OpAccessChain %p_address %push %zero
%addr = OpLoad %address %field
%dst = OpBitcast %p_word %addr
OpStore %dst %value Aligned 4
OpReturn
OpFunctionEnd
"""


def snapshot(root):
    """Include directories, bytes and modification times, so even journal upkeep is observable."""
    return {
        p.relative_to(root): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).digest() if p.is_file() else None)
        for p in root.rglob("*")
    }


def check(exe, root, device, require_hit):
    shaders = root / "shaders"
    shaders.mkdir()
    module = assemble(SHADER, "vulkan1.0")
    (shaders / "authored.spv").write_bytes(module)
    index = {"authored": {"stage": "cs", "fssm_sha256": hashlib.sha256(module).hexdigest(), "local_size": [1, 1, 1]}}
    (shaders / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")
    cache = root / "cache"
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("LEONIDA_")}
    env.update(LEONIDA_SHADERS=str(shaders), LEONIDA_PIPELINE_CACHE_DIR=str(cache))
    if device:
        env["LEONIDA_VK_DEVICE"] = device

    def run(*args):
        return subprocess.run([str(exe), "--precompile-pipelines", *args], cwd=root, env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=60)

    def verify(*args):
        before = snapshot(root)
        result = run("--verify", *args)
        if snapshot(root) != before:
            raise RuntimeError("verification changed the temporary library or cache tree")
        return result

    def expect(result, code, summary):
        if result.returncode != code or summary not in result.stderr:
            raise RuntimeError(f"expected exit {code} and {summary!r}, got {result.returncode}\n{result.stderr}")

    missing = verify()
    if missing.returncode == 1 and any(reason in missing.stderr for reason in (
            "the Vulkan device could not be created", "the runtime builds no compute pipeline on this device",
            "requires enabled pipelineCreationCacheControl")):
        print("SKIP: device cannot run the journal probe")
        return 77
    expect(missing, 2, "selected=1 verified=0 missing_or_unusable=1 unverified=0 errors=0")
    expect(verify("--filter", "no-such-name"), 2, "selected=0 verified=0")
    env["LEONIDA_PIPELINE_JOURNAL"] = "0"
    expect(verify(), 1, "verification requires an enabled local pipeline journal")
    del env["LEONIDA_PIPELINE_JOURNAL"]
    warm = run("--threads", "1")
    expect(warm, 0, "0 failed")
    files = list((cache / "pipeline_cache.d").glob("cs-*.bin"))
    if not files and "no cache data for authored" in warm.stderr and not require_hit:
        print("SKIP: the driver returned no opaque cache payload")
        return 77
    if len(files) != 1:
        raise RuntimeError(f"expected one published cache file, found {len(files)}\n{warm.stderr}")
    entry, = files
    # Upkeep would evict the entry under this budget, remove the stale staging file or touch a hit.
    stale = entry.with_name(entry.name + ".tmp.1.1")
    stale.write_bytes(b"stale")
    for path in (entry, stale):
        os.utime(path, (1000000000, 1000000000))
    env["LEONIDA_PIPELINE_JOURNAL_BYTES"] = "1"
    hit = verify()
    confirmed = hit.returncode == 0
    if confirmed or require_hit:
        expect(hit, 0, "selected=1 verified=1 missing_or_unusable=0 unverified=0 errors=0")
    else:
        expect(hit, 2, "selected=1 verified=0 missing_or_unusable=0 unverified=1 errors=0")

    # A second entry changes the actual module, so the first entry's hit cannot satisfy the full selection.
    # Vary the literal to avoid reusing a previous run's store module.
    value = int.from_bytes(hashlib.sha256(root.name.encode()).digest()[:4], "little")
    changed = assemble(STORE_SHADER.format(value=value), "vulkan1.2")
    (shaders / "second.spv").write_bytes(changed)
    index["second"] = {"stage": "cs", "fssm_sha256": hashlib.sha256(changed).hexdigest(), "local_size": [1, 1, 1]}
    (shaders / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")
    expect(verify(), 2, f"selected=2 verified={int(confirmed)} missing_or_unusable=1 "
                        f"unverified={int(not confirmed)} errors=0")
    del env["LEONIDA_PIPELINE_JOURNAL_BYTES"]
    expect(run("--filter", "second", "--threads", "1"), 0, "0 failed")
    second_files = set(entry.parent.glob("cs-*.bin")) - {entry}
    if len(second_files) != 1:
        raise RuntimeError("expected a second local journal file for the store shader")
    second, = second_files
    second.write_bytes(entry.read_bytes())
    expect(verify("--filter", "second"), 2, "selected=1 verified=0 missing_or_unusable=0 unverified=1 errors=0")
    (shaders / "second.spv").unlink()
    expect(verify("--limit", "1", "--smallest-first", "--min-bytes", "1"), 2,
           f"selected=1 verified={int(confirmed)} missing_or_unusable=0 unverified={int(not confirmed)} errors=1")

    env["LEONIDA_SKIP_COMPUTE"] = "second"
    expect(verify("--skip-listed"), 0 if confirmed else 2,
           f"selected=1 verified={int(confirmed)} missing_or_unusable=0 unverified={int(not confirmed)} "
           "errors=0 filtered=0 not_buildable=0 skipped_listed=1")
    expect(verify(), 2, "errors=1")
    del env["LEONIDA_SKIP_COMPUTE"]

    # The identical entry in an external seed does not certify a missing local journal file.
    seed = root / "seed"
    seed.mkdir()
    original = entry.rename(seed / entry.name)
    env["LEONIDA_PIPELINE_JOURNAL_SEED_DIR"] = str(seed)
    expect(verify("--filter", "authored"), 2, "selected=1 verified=0 missing_or_unusable=1 unverified=0 errors=0")
    entry.write_bytes(original.read_bytes()[:32])  # matching header alone cannot confirm reuse
    expect(verify("--filter", "authored"), 2, "selected=1 verified=0 missing_or_unusable=1 unverified=0 errors=0")
    (shaders / "authored.spv").write_bytes(b"bad!")
    expect(verify("--filter", "authored"), 2, "selected=1 verified=0 missing_or_unusable=0 unverified=0 errors=1")
    print("Verified read-only journal checks; driver reported application-cache hit:", confirmed)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="built gta6rt.exe")
    parser.add_argument("--device", help="optional LEONIDA_VK_DEVICE selector")
    parser.add_argument("--require-hit", action="store_true", help="require a driver-confirmed hit after warming")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="leonida-precompile-verify-") as tmp:
        sys.exit(check(args.executable.resolve(strict=True), Path(tmp), args.device, args.require_hit))

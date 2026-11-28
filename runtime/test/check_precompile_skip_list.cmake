# gta6rt --precompile-pipelines and LEONIDA_SKIP_COMPUTE (shims/d3d12_precompile.h): the play-time skip list does not
# keep programs out of the pipeline journal; --skip-listed does, and the closing line counts both.
#   cmake -DGTA6RT=<gta6rt.exe> [-DRUNNER=<wine>] -DSPV_A=<compute .spv> -DSPV_B=<another compute .spv>
#         -DTEST_DIR=<scratch folder> -P check_precompile_skip_list.cmake
# A library of two synthetic compute entries, one of them named by LEONIDA_SKIP_COMPUTE, is precompiled three times:
#   1. into an empty journal with the list set and no flag: both entries are built;
#   2. into another empty journal with --skip-listed: only the unlisted one is, and the listed one is reported;
#   3. into the first journal again: both are journaled already, nothing is built.
# The verdicts are read from the output and the journal folders, since under Wine an exit status alone can look fine.
cmake_minimum_required(VERSION 3.24)

foreach(var GTA6RT SPV_A SPV_B TEST_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "check_precompile_skip_list.cmake: -D${var}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}/shaders")
# Two different modules: entries with the same module and layout share one journal file.
file(COPY_FILE "${SPV_A}" "${TEST_DIR}/shaders/precompile_kept.spv")
file(COPY_FILE "${SPV_B}" "${TEST_DIR}/shaders/precompile_listed.spv")
# The runtime's library takes an entry only with a program hash; these two name no real program.
set(meta "\"stage\":\"cs\",\"bindings\":[],\"push_constant_bytes\":144,\"local_size\":[1,1,1],\"wave64\":false")
string(REPEAT "a" 64 hash_kept)
string(REPEAT "b" 64 hash_listed)
file(WRITE "${TEST_DIR}/shaders/index.json"
     "{\"precompile_kept\":{\"fssm_sha256\":\"${hash_kept}\",${meta}},"
     "\"precompile_listed\":{\"fssm_sha256\":\"${hash_listed}\",${meta}}}\n")

# Only what this test sets: a journal setting or helper from the caller's environment would change the counts.
foreach(var LEONIDA_PIPELINE_JOURNAL LEONIDA_PIPELINE_CACHE_BYTES LEONIDA_PIPELINE_JOURNAL_BYTES
            LEONIDA_PIPELINE_JOURNAL_SEED_DIR LEONIDA_COMPILE_HELPER LEONIDA_LAZY_COMPUTE)
  unset(ENV{${var}})
endforeach()
file(TO_NATIVE_PATH "${TEST_DIR}/shaders" shaders)
set(ENV{LEONIDA_SHADERS} "${shaders}")
set(ENV{LEONIDA_SKIP_COMPUTE} "precompile_listed")

set(failures "")
set(all_output "")
# Runs the precompile with the journal in <cache> and the extra arguments; sets `out` and `code` in the caller.
function(precompile label cache)
  file(TO_NATIVE_PATH "${TEST_DIR}/${cache}" cache_dir)
  set(ENV{LEONIDA_PIPELINE_CACHE_DIR} "${cache_dir}")
  execute_process(COMMAND ${RUNNER} "${GTA6RT}" --precompile-pipelines --threads 1 ${ARGN}
                  WORKING_DIRECTORY "${TEST_DIR}"
                  TIMEOUT 90
                  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
  string(REPLACE "\r\n" "\n" output "\n${output}\n")
  set(out "${output}" PARENT_SCOPE)
  set(code "${result}" PARENT_SCOPE)
  set(all_output "${all_output}--- ${label} (exit ${result}) ---${output}" PARENT_SCOPE)
endfunction()
# Functions, not macros: a macro would substitute the patterns' text and undo their backslash escapes.
function(expect label pattern)
  if(NOT out MATCHES "${pattern}")
    set(failures "${failures}${label}: no match for '${pattern}'\n" PARENT_SCOPE)
  endif()
endfunction()
function(expect_not label pattern)
  if(out MATCHES "${pattern}")
    set(failures "${failures}${label}: unexpected match for '${pattern}'\n" PARENT_SCOPE)
  endif()
endfunction()
function(expect_exit label want)
  if(NOT "${code}" STREQUAL "${want}")
    set(failures "${failures}${label}: exit ${code}, not ${want}\n" PARENT_SCOPE)
  endif()
endfunction()
function(journal_files cache var)
  file(GLOB files "${TEST_DIR}/${cache}/pipeline_cache.d/cs-*.bin")
  list(LENGTH files count)
  set(${var} ${count} PARENT_SCOPE)
endfunction()

# 1. The list is set, no flag: both entries are built and nothing is reported as left out by the list.
precompile("list set, no flag" cache_all)
if(code STREQUAL "1" AND out MATCHES "the Vulkan device could not be created|the runtime builds no compute pipeline")
  message("SKIP: no Vulkan device for the precompile here${all_output}")
  return()
endif()
expect_exit("list set, no flag" 0)
expect("list set, no flag" "\n\\[precompile\\] LEONIDA_SKIP_COMPUTE names 1 of the entries: built anyway")
expect("list set, no flag" "\n\\[precompile\\] 2 compute entries to build ")
expect_not("list set, no flag" "precompile_listed: not built")
expect("list set, no flag" "\n\\[precompile\\] done: 2 built \\([0-9]+ compiled, [0-9]+ from a cache\\), 0 journaled already, 0 being built by another process, 0 failed; left out: 0 of 1 listed in LEONIDA_SKIP_COMPUTE, 0 not buildable, 0 filtered out; in [0-9]+ s\n")
# A driver that returns no cache data leaves no journal file; then step 3 has nothing to find.
set(journaled TRUE)
if(out MATCHES "compiled \\(not journaled\\)")
  set(journaled FALSE)
  message("note: the driver gave no cache data, so the journal-file checks are left out")
else()
  journal_files(cache_all count)
  if(NOT count EQUAL 2)
    string(APPEND failures "list set, no flag: ${count} journal files, not 2\n")
  endif()
endif()

# 2. --skip-listed: the listed entry is left out and named, the other one is built.
precompile("--skip-listed" cache_listed --skip-listed)
expect_exit("--skip-listed" 0)
expect("--skip-listed" "\n\\[precompile\\] precompile_listed: not built \\(LEONIDA_SKIP_COMPUTE, --skip-listed\\)\n")
expect("--skip-listed" "\n\\[precompile\\] 1 compute entries to build \\([^)]*, 1 left out by --skip-listed, 0 not buildable\\)")
expect("--skip-listed" "\n\\[precompile\\] done: 1 built \\([0-9]+ compiled, [0-9]+ from a cache\\), 0 journaled already, 0 being built by another process, 0 failed; left out: 1 of 1 listed in LEONIDA_SKIP_COMPUTE \\(--skip-listed\\), 0 not buildable, 0 filtered out; in [0-9]+ s\n")
if(journaled)
  journal_files(cache_listed count)
  if(NOT count EQUAL 1)
    string(APPEND failures "--skip-listed: ${count} journal files, not 1\n")
  endif()
endif()

# 3. The first journal again: both entries are there, nothing is built.
if(journaled)
  precompile("again" cache_all)
  expect_exit("again" 0)
  expect("again" "\n\\[precompile\\] done: 0 built \\(0 compiled, 0 from a cache\\), 2 journaled already, 0 being built by another process, 0 failed; left out: 0 of 1 listed in LEONIDA_SKIP_COMPUTE, 0 not buildable, 0 filtered out; in [0-9]+ s\n")
endif()

if(failures)
  message(FATAL_ERROR "gta6rt --precompile-pipelines and LEONIDA_SKIP_COMPUTE:\n${failures}${all_output}")
endif()
message("precompile and LEONIDA_SKIP_COMPUTE: built regardless of the list, left out with --skip-listed${all_output}")

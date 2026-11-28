# The cache vk_pipeline_check saves must hit on the runtime's device: the checker compiles one synthetic compute
# shader into a cache, then pipeline_cache_seed_test creates the same pipeline on the runtime's device from it.
#
# The checker runs as a copy with the test program's file name: a driver can make the executable's file name part of
# the cache key (docs/pipeline_check.md, "The cache and the runtime's device"), and this test is about the device.
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}/checker")
file(COPY_FILE "${SHADER}" "${TEST_DIR}/seed.spv")
file(WRITE "${TEST_DIR}/index.json" "{\"seed\":{\"stage\":\"cs\",\"bindings\":[],\"push_constant_bytes\":144,"
                                    "\"local_size\":[1,1,1],\"wave64\":false}}\n")
get_filename_component(test_name "${TEST}" NAME)
set(checker "${TEST_DIR}/checker/${test_name}")
file(COPY_FILE "${CHECKER}" "${checker}")

execute_process(COMMAND ${RUNNER} "${checker}" "${TEST_DIR}" "${TEST_DIR}/pipeline_cache.bin" --jobs 1
                RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 120)
if(NOT log MATCHES "cs pipelines: 1 ok, 0 failed, 0 skipped" OR NOT EXISTS "${TEST_DIR}/pipeline_cache.bin")
  if(log MATCHES "no Vulkan physical device|vulkan-1|libvulkan|cs pipelines: 0 ok, 0 failed, 1 skipped")
    message(STATUS "SKIP: vk_pipeline_check built no pipeline here\n${log}")
    return()
  endif()
  message(FATAL_ERROR "vk_pipeline_check did not build the pipeline (${code})\n${log}")
endif()

execute_process(COMMAND ${RUNNER} "${TEST}" "${TEST_DIR}/seed.spv" "${TEST_DIR}/pipeline_cache.bin"
                RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 120)
message(STATUS "${log}")
if(code STREQUAL "77")
  return()  # the test printed its own SKIP line
endif()
if(NOT code STREQUAL "0")
  message(FATAL_ERROR "pipeline_cache_seed_test failed (${code})")
endif()

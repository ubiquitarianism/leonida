foreach(mode args no_args fail_init)
  set(args)
  if(mode STREQUAL "args")
    set(args hello "two words")
  elseif(mode STREQUAL "fail_init")
    set(args fail-init)
  endif()
  # The native run is the control: it shows that what the fixture expects at process attach is what Microsoft's UCRT
  # does. Through an emulator (RUNNER, the cross build's Wine) the C runtime is the emulator's own, and Wine's ucrtbase
  # has __argc, __argv and __wargv set before process attach (wine-staging 11.21), so the control cannot hold there:
  # it is reported and not required. The run through the loader below is checked on every host.
  if(NOT mode STREQUAL "fail_init")
    execute_process(COMMAND ${RUNNER} "${GUEST}" ${args} TIMEOUT 90
      RESULT_VARIABLE native_code OUTPUT_VARIABLE native_output ERROR_VARIABLE native_error)
    if(NOT native_code STREQUAL "0" OR NOT native_output MATCHES "tiny_startup_cmdline: PASS")
      if("${RUNNER}" STREQUAL "")
        message(FATAL_ERROR "Native startup command line (${mode}) returned ${native_code}\n${native_output}\n${native_error}")
      endif()
      message(STATUS "SKIP the native control (${mode}): it returned ${native_code} through ${RUNNER}, "
        "whose C runtime is not Microsoft's")
    endif()
  endif()
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" --run "${GUEST}" -- ${args}
    TIMEOUT 90
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(mode STREQUAL "fail_init")
    if(NOT code STREQUAL "1" OR NOT error MATCHES "DllMain returned FALSE" OR output MATCHES "running ")
      message(FATAL_ERROR "Failed attach returned ${code}\n${output}\n${error}")
    endif()
    if(NOT output MATCHES "2 module\\(s\\) mapped, [0-9]+ unimplemented import\\(s\\)"
        OR NOT output MATCHES "unimplemented by return policy: hresult=[0-9]+ other=[0-9]+")
      message(FATAL_ERROR "Failed attach lost the import binding report\n${output}\n${error}")
    endif()
  elseif(NOT code STREQUAL "0" OR NOT output MATCHES "tiny_startup_cmdline: PASS"
      OR NOT output MATCHES "game exited with code 0")
    message(FATAL_ERROR "Startup command line (${mode}) returned ${code}\n${output}\n${error}")
  endif()
endforeach()

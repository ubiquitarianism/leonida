# Resolves -DLEONIDA_XMA2_DECODER_LIB=<path> for runtime/CMakeLists.txt.
# Kept in its own file so tests/test_cmake_xma2_prebuilt.py can run it with `cmake -P`.
#
# When LEONIDA_XMA2_DECODER_LIB is set, the runtime links that prebuilt static library directly and skips
# running cargo at configure time (useful for offline builds and Linux cross-builds where the distribution's
# rustc does not ship the x86_64-pc-windows-msvc standard library). A value that does not name an existing
# file is a configure error rather than a silent fallback to muted XMA voices.
function(leonida_resolve_prebuilt_xma2_decoder lib_input base_dir out_var)
  if(NOT lib_input)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()
  if(IS_ABSOLUTE "${lib_input}")
    set(_resolved "${lib_input}")
  else()
    cmake_path(ABSOLUTE_PATH lib_input BASE_DIRECTORY "${base_dir}" NORMALIZE OUTPUT_VARIABLE _resolved)
  endif()
  if(EXISTS "${_resolved}" AND NOT IS_DIRECTORY "${_resolved}")
    set(${out_var} "${_resolved}" PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "LEONIDA_XMA2_DECODER_LIB does not point to an existing file: ${_resolved} "
      "(pass the path to leonida_xma2_decoder.lib or leave it empty to build with cargo)")
  endif()
endfunction()

# Resolves -DLEONIDA_OPUS_DECODER_LIB=<path> for runtime/CMakeLists.txt.
#
# When LEONIDA_OPUS_DECODER_LIB is set, the runtime links that prebuilt static library directly and
# skips running cargo at configure time. A value that does not name an existing file is a configure
# error rather than a silent fallback to silent XAPU output.
function(leonida_resolve_prebuilt_opus_decoder lib_input base_dir out_var)
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
      "LEONIDA_OPUS_DECODER_LIB does not point to an existing file: ${_resolved} "
      "(pass the path to leonida_opus_decoder.lib or leave it empty to build with cargo)")
  endif()
endfunction()

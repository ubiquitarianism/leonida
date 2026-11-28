# Redirect the loader's stdout so the console paths (GAME:, the overlay and T:'s folder in it) must survive as UTF-8.
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --game-dir "game-Игра" --overlay "overlay-保存"
          --run "${GUEST}" -- hello 42
  TIMEOUT 90 ENCODING UTF-8
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT code STREQUAL "0" OR
   NOT output MATCHES "console paths: GAME: -> [^\n]*game-Игра, writes -> [^\n]*overlay-保存, T: -> [^\n]*overlay-保存[\\/]\\.gta6rt-temp[\\/][^\n]*pid[0-9]+\r?\n")
  message(FATAL_ERROR "Runtime did not report the Unicode paths (exit ${code})\n${output}\n${error}")
endif()
if(NOT output MATCHES "(^|\n)tiny_game: OK(\r?\n|$)" OR
   NOT output MATCHES "(^|\n)game exited with code 0(\r?\n|$)")
  message(FATAL_ERROR "Synthetic guest did not finish successfully\n${output}\n${error}")
endif()

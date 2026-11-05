function(checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE result
    OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer command failed (${result}): ${ARGV}\n${out}\n${err}")
  endif()
endfunction()

set(prefix "${PREPROCESS_BUILD_DIR}/installed-test/prefix")
set(consumer "${PREPROCESS_BUILD_DIR}/installed-test/consumer")
checked("${CMAKE_COMMAND}" --install "${PREPROCESS_BUILD_DIR}" --prefix "${prefix}")
checked("${CMAKE_COMMAND}" -S "${PREPROCESS_SOURCE_DIR}/tests/consumer"
  -B "${consumer}" -G Ninja
  "-DCMAKE_CXX_COMPILER=${PREPROCESS_COMPILER}"
  "-DCMAKE_BUILD_TYPE=${PREPROCESS_CONFIG}"
  "-DCMAKE_PREFIX_PATH=${prefix}\\;${PREPROCESS_ANNDATAPP_DIR}/../../.."
  "-DPREPROCESS_SANITIZERS=${PREPROCESS_SANITIZERS}")
checked("${CMAKE_COMMAND}" --build "${consumer}" --parallel 2)
checked("${consumer}/consumer")
checked("${consumer}/consumer_again")

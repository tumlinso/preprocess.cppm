cmake_minimum_required(VERSION 3.28)
project(preprocess VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "Use Clang 18 or newer with matching clang-scan-deps.")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
  message(FATAL_ERROR "Use Ninja for C++20 module dependency scanning.")
endif()
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  string(REGEX MATCH "^[0-9]+" _clang_major "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
    NAMES clang-scan-deps-${_clang_major} clang-scan-deps REQUIRED)
endif()
find_package(anndatapp 0.1 CONFIG REQUIRED)

option(PREPROCESS_BUILD_TESTS "Build native and installed-consumer tests" ${PROJECT_IS_TOP_LEVEL})
option(PREPROCESS_BUILD_QUALIFICATION "Build independent Python reference driver" OFF)
option(PREPROCESS_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

add_library(preprocess)
add_library(preprocess::preprocess ALIAS preprocess)
set_target_properties(preprocess PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME preprocess_binary)
target_compile_features(preprocess PUBLIC cxx_std_20)
target_sources(preprocess
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
    FILES src/preprocess.cppm
  PRIVATE src/preprocess.cpp src/hvg.cpp)
target_link_libraries(preprocess PUBLIC anndatapp::anndatapp)
target_compile_options(preprocess PRIVATE -Wall -Wextra -Wpedantic)

if(PREPROCESS_ENABLE_SANITIZERS)
  target_compile_options(preprocess PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(preprocess PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS preprocess EXPORT preprocessTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/preprocess/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT preprocessTargets NAMESPACE preprocess:: CXX_MODULES_DIRECTORY cxx-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/preprocess)

configure_package_config_file(cmake/preprocessConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/preprocessConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/preprocess)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/preprocessConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/preprocessConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/preprocessConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/preprocess)
install(FILES LICENSE THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/preprocess)

add_executable(preprocess_example examples/basic.cpp)
target_link_libraries(preprocess_example PRIVATE preprocess::preprocess)

if(PREPROCESS_BUILD_TESTS)
  enable_testing()
  foreach(suite IN ITEMS numerics hvg)
    add_executable(test_${suite} tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE preprocess::preprocess)
    add_test(NAME ${suite} COMMAND test_${suite})
  endforeach()
  add_test(NAME example COMMAND preprocess_example)
  add_test(NAME installed-consumer COMMAND ${CMAKE_COMMAND}
    -DPREPROCESS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DPREPROCESS_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DPREPROCESS_COMPILER=${CMAKE_CXX_COMPILER}
    -DPREPROCESS_CONFIG=${CMAKE_BUILD_TYPE}
    -DPREPROCESS_ANNDATAPP_DIR=${anndatapp_DIR}
    -DPREPROCESS_SANITIZERS=${PREPROCESS_ENABLE_SANITIZERS}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake)
endif()

if(PREPROCESS_BUILD_QUALIFICATION)
  add_executable(preprocess_interop_driver tests/interop_driver.cpp)
  target_link_libraries(preprocess_interop_driver PRIVATE preprocess::preprocess)
  add_executable(preprocess_benchmark benchmarks/qualification.cpp)
  target_link_libraries(preprocess_benchmark PRIVATE preprocess::preprocess)
endif()

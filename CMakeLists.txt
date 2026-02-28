cmake_minimum_required(VERSION 3.20)

project(
    treeroot
    VERSION 0.1.0
    DESCRIPTION "Experimental Monadic Datalog Engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
find_program(CLANG_FORMAT_EXE NAMES clang-format-22 clang-format)

if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/headers/*.hpp
    )
    add_custom_target(format COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES})
    message(STATUS "Found clang-format: ${CLANG_FORMAT_EXE}")
else()
    message(WARNING "clang-format not found")
endif()

find_package(GTest REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

add_library(treeroot INTERFACE)
target_include_directories(treeroot INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/headers
)

add_executable(treeroot_tests
    src/datalog_tests.cpp
)
target_link_libraries(treeroot_tests PRIVATE treeroot GTest::GTest GTest::Main benchmark)

# add_executable(treeroot_benchmarks
#     src/datalog_benchmarks.cpp
# )
# target_link_libraries(treeroot_benchmarks PRIVATE treeroot benchmark)

# enable_testing()
# gtest_discover_tests(treeroot_tests)

# enable_benchmarking()
# add_benchmarks(treeroot_benchmarks)
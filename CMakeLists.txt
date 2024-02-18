cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 11)
project(calibration_in_llm)

include(FetchContent)
find_package(ZLIB REQUIRED)

# Declare the external project
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        0353a1840134b24b07ab61fd4490192f28c4db6b
)

# Fetch and populate the content of the external project
FetchContent_MakeAvailable(llama_cpp)

# Build for extract_probabilities.cpp
set(TARGET extract_probabilities)
add_executable(${TARGET} src/commands/extract_probabilities.cpp
    src/utils/llama_cpp_helper.cpp
    src/utils/input_iterator.cpp
    src/utils/result_writer.cpp
    src/utils/parse_custom_params.cpp
    src/utils/position_result/position_result.cpp
    src/utils/position_result/position_full_result.cpp
    src/utils/position_result/position_top_result.cpp)
target_include_directories(${TARGET} PRIVATE include)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Needed for metal on macOS
configure_file(
    ${llama_cpp_SOURCE_DIR}/ggml-metal.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/build/ggml-metal.metal
    COPYONLY
)

# Include test subdirectory
add_subdirectory(tests)
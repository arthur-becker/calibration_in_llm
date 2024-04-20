cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 17)
project(calibration_in_llm)

include(FetchContent)
find_package(ZLIB REQUIRED)


function(get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang")
        set(C_FLAGS   -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if (
            (CCID STREQUAL "Clang"      AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR
            (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0)
        )
            set(C_FLAGS ${C_FLAGS} -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 7.1.0)
            set(CXX_FLAGS ${CXX_FLAGS} -Wno-format-truncation)
        endif()
        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            set(CXX_FLAGS ${CXX_FLAGS} -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()


if (LLAMA_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_HEADERS_CUDA ggml-cuda.h)
        set(GGML_SOURCES_CUDA ggml-cuda.cu)

        add_compile_definitions(GGML_USE_CUBLAS)
#        if (LLAMA_CUDA_CUBLAS)
#            add_compile_definitions(GGML_CUDA_CUBLAS)
#        endif()
        if (LLAMA_CUDA_FORCE_DMMV)
            add_compile_definitions(GGML_CUDA_FORCE_DMMV)
        endif()
        if (LLAMA_CUDA_FORCE_MMQ)
            add_compile_definitions(GGML_CUDA_FORCE_MMQ)
        endif()
        add_compile_definitions(GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
        add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
        if (DEFINED LLAMA_CUDA_DMMV_Y)
            add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_DMMV_Y}) # for backwards compatibility
        endif()
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            add_compile_definitions(GGML_CUDA_F16)
        endif()
        add_compile_definitions(K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})
        add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${LLAMA_CUDA_PEER_MAX_BATCH_SIZE})

        if (LLAMA_STATIC)
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 52 == lowest CUDA 12 standard
        # 60 == f16 CUDA intrinsics
        # 61 == integer CUDA intrinsics
        # 70 == compute capability at which unrolling a loop in mul_mat_q kernels is faster
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70") # needed for f16 CUDA intrinsics
        else()
            set(CMAKE_CUDA_ARCHITECTURES "52;61;70") # lowest CUDA 12 standard + lowest for integer intrinsics
            #set(CMAKE_CUDA_ARCHITECTURES "") # use this to compile much faster, but only F16 models work
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (LLAMA_CUBLAS)
    set(CUDA_FLAGS ${CXX_FLAGS} -use_fast_math)
    if (NOT MSVC)
        set(CUDA_FLAGS ${CUDA_FLAGS} -Wno-pedantic)
    endif()

    if (LLAMA_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            set(NVCC_CMD ${NVCC_CMD} -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(JOIN GF_CXX_FLAGS " " CUDA_CXX_FLAGS)  # pass host compiler flags as a single argument
        if (NOT CUDA_CXX_FLAGS STREQUAL "")
            set(CUDA_FLAGS ${CUDA_FLAGS} -Xcompiler ${CUDA_CXX_FLAGS})
        endif()
    endif()

    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

# Llama.cpp
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        0353a1840134b24b07ab61fd4490192f28c4db6b
)
FetchContent_MakeAvailable(llama_cpp)

# yaml-cpp
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

# Build for extract_probabilities.cpp
set(TARGET extract_probabilities)
add_executable(${TARGET} src/commands/extract_probabilities.cpp
    src/utils/llama_cpp_helper.cpp
    src/utils/input_iterator.cpp
    src/utils/result_writer.cpp
    src/utils/parse_custom_params.cpp
    src/utils/position_result/position_result.cpp
    src/utils/position_result/position_full_result.cpp
    src/utils/position_result/position_top_result.cpp
    src/utils/softmax.cpp
    src/utils/endianness.cpp)
target_include_directories(${TARGET} PRIVATE include)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ZLIB::ZLIB yaml-cpp::yaml-cpp ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)



# Build for extract_probabilities.cpp
set(TARGET check_integrity)
add_executable(${TARGET} src/commands/check_integrity.cpp
    src/utils/result_writer.cpp
    src/utils/result_reader.cpp
    src/utils/position_result/position_result.cpp
    src/utils/position_result/position_full_result.cpp
    src/utils/position_result/position_top_result.cpp)
target_include_directories(${TARGET} PRIVATE include)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Build for generate_test_data.cpp
set(TARGET generate_mock_data)
add_executable(${TARGET} src/commands/generate_mock_data.cpp
    src/utils/result_writer.cpp
    src/utils/parse_custom_params.cpp
    src/utils/position_result/position_result.cpp
    src/utils/position_result/position_full_result.cpp
    src/utils/position_result/position_top_result.cpp
    src/utils/softmax.cpp
    src/utils/endianness.cpp)
target_include_directories(${TARGET} PRIVATE include)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ZLIB::ZLIB yaml-cpp::yaml-cpp ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Needed for metal on macOS
configure_file(
    ${llama_cpp_SOURCE_DIR}/ggml-metal.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/build/ggml-metal.metal
    COPYONLY
)

# Include test subdirectory
add_subdirectory(tests)

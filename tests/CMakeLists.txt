cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(ZLIB REQUIRED)

# Add the directory containing your source files
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# Add your test executable and link it with the source files
add_executable(result_writer_test result_writer_test.cpp
    ../src/utils/result_writer.cpp 
    ../src/utils/result_reader.cpp
    ../src/utils/position_result/position_result.cpp
    ../src/utils/position_result/position_full_result.cpp
    ../src/utils/position_result/position_top_result.cpp)


add_executable(test_top_result_data_structure test_top_result_data_structure.cpp
    ../src/utils/position_result/position_result.cpp
    ../src/utils/position_result/position_top_result.cpp)

add_executable(test_checksum test_checksum.cpp
    ../src/utils/position_result/position_result.cpp
    ../src/utils/position_result/position_full_result.cpp
    ../src/utils/position_result/position_top_result.cpp)

add_executable(test_custom_params_parser test_custom_params_parser.cpp
    ../src/utils/parse_custom_params.cpp)

target_link_libraries(result_writer_test PRIVATE ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_top_result_data_structure PRIVATE ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_checksum PRIVATE ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_custom_params_parser PRIVATE ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})

# Register the test
enable_testing()
add_test(NAME TopResultDataStructureTest COMMAND test_top_result_data_structure)
add_test(NAME ChecksumTest COMMAND test_checksum)
add_test(NAME ResultWriterTest COMMAND result_writer_test)
add_test(NAME CustomParamsParserTest COMMAND test_custom_params_parser)
cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(ZLIB REQUIRED)

# Add the directory containing your source files
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# Add your test executable and link it with the source files
add_executable(result_writer_test result_writer_test.cpp ../src/utils/result_writer.cpp ../src/utils/result_reader.cpp)

target_link_libraries(result_writer_test PRIVATE ZLIB::ZLIB ${CMAKE_THREAD_LIBS_INIT})

# Register the test
enable_testing()
add_test(NAME ResultWriterTest COMMAND result_writer_test)
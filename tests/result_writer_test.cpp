#include <cstdio>
#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "utils/result_writer.h"
#include "utils/result_reader.h"
#include "utils/position_result/position_result.h"
#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"



template <typename T>
void test_result_writer(T mock_data[], uint16_t mock_data_size){
    static_assert(std::is_base_of<PositionResult, T>::value, "T must be a derived class of PositionResult");

    std::string filename = "test_result_writer.txt";

    // If file exists, delete it
    bool file_exists = std::ifstream(filename).good();
    if (file_exists){
        std::remove(filename.c_str());
        printf("Removed file %s\n", filename.c_str());
    }

    // Create ResultWriter
    ResultWriter<T> writer(filename);
    writer.openFile();
    for (uint16_t i = 0; i < mock_data_size; i++){
        T data = mock_data[i];
        writer.addPositionResult(data);
    }
    writer.writeAndClear();
    writer.closeFile();

    // Bytes
    uint32_t bytes_number = 0;
    for (uint16_t i = 0; i < mock_data_size; i++){
        T data = mock_data[i];
        bytes_number += data.countBytes() + sizeof(data.getChecksum());
    }

    // Count bytes of filename
    std::ifstream
        file(filename, std::ios::binary | std::ios::ate);
    uint32_t file_size = file.tellg();
    file.close();
    assert(bytes_number == file_size);

    // Read the file
    ResultReader<T> reader;
    std::vector<T> read_results = reader.read_result(filename);
    assert(read_results.size() == mock_data_size);

    // Compare the results
    for (int i = 0; i < mock_data_size; i++){
        assert(read_results[i].getChecksum() == mock_data[i].getChecksum());
    }
}

int main() {
    // Mock data
    float token_data1[] = {0.1, 0.2, 0.6, 0.3};
    float token_data2[] = {0.2, 0.5, 0.3, 0.2, 0.8, 0.1};

    // PositionFullResult
    PositionFullResult full_result1(
        std::vector<float>(token_data1, token_data1 + 4),
        1
    );
    PositionFullResult full_result2(
        std::vector<float>(token_data2, token_data2 + 6),
        2
    );
    PositionFullResult full_results[] = {full_result1, full_result1};
    uint16_t full_results_size = sizeof(full_results) / sizeof(full_results[0]);
    test_result_writer(full_results, full_results_size);
    printf("ResultWriter test for `PositionFullResult` passed!\n");

    // PositionTopResult
    uint16_t k = 3; // top-k
    PositionTopResult top_result1(
        std::vector<float>(token_data1, token_data1 + 4),
        1,
        k
    );
    PositionTopResult top_result2(
        std::vector<float>(token_data2, token_data2 + 6),
        2,
        k
    );
    PositionTopResult top_results[] = {top_result1, top_result2};
    uint16_t top_results_size = 2;
    test_result_writer(top_results, top_results_size);
    printf("ResultWriter test for `PositionTopResult` passed!\n");
    
    return 0;
}
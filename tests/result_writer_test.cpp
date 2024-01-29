#include "utils/result_writer.h"
#include "utils/result_reader.h"
#include <cstdio>
#include <cassert>
#include <vector>
#include <iostream>

void test_checksum(){
    // Equal token data & correct token -> equal checksums
    float token_data1[] = {0.1, 0.2, 0.7};
    PositionResult result1(
        std::vector<float>(token_data1, token_data1 + 3),
        1
    );
    float token_data2[] = {0.1, 0.2, 0.7};
    PositionResult result2(
        std::vector<float>(token_data2, token_data2 + 3),
        1
    );
    assert(result1.getChecksum() == result2.getChecksum());

    // Different correct tokens -> different checksums
    float token_data3[] = {0.1, 0.2, 0.7};
    PositionResult result3(
        std::vector<float>(token_data3, token_data3 + 3),
        0
    );
    assert(result1.getChecksum() != result3.getChecksum());

    // Different token data -> different checksums
    float token_data4[] = {0.1, 0.3, 0.6};
    PositionResult result4(
        std::vector<float>(token_data4, token_data4 + 3),
        1
    );
    assert(result1.getChecksum() != result4.getChecksum());

    // Different token data & different correct token -> different checksums
    float token_data5[] = {0.1, 0.5, 0.4};
    PositionResult result5(
        std::vector<float>(token_data5, token_data5 + 3),
        2
    );
    assert(result1.getChecksum() != result5.getChecksum());

    printf("Checksum test passed!\n");
}

void test_result_writer(){
    std::string filename = "test_result_writer.txt";

    // If file exists, delete it
    bool file_exists = std::ifstream(filename).good();
    if (file_exists){
        std::remove(filename.c_str());
        printf("Removed file %s\n", filename.c_str());
    }

    // Mock data
    float token_data1[] = {0.1, 0.2, 0.6};
    PositionResult result1(
        std::vector<float>(token_data1, token_data1 + 3),
        1
    );
    float token_data2[] = {0.2, 0.5};
    PositionResult result2(
        std::vector<float>(token_data2, token_data2 + 2),
        2
    );
    PositionResult results[] = {result1, result2};

    // Create ResultWriter
    ResultWriter writer(filename);
    writer.openFile();
    writer.addPositionResult(result1);
    writer.addPositionResult(result2);
    writer.writeAndClear();
    writer.closeFile();

    // Read the file
    ResultReader reader;
    std::vector<PositionResult> read_results = reader.read_result(filename);
    assert(read_results.size() == 2);
    PositionResult read_result1 = read_results[0];
    PositionResult read_result2 = read_results[1];

    // Compare the results
    assert(read_result1.getChecksum() == result1.getChecksum());
    assert(read_result2.getChecksum() == result2.getChecksum());

    printf("ResultWriter test passed!\n");
}

int main() {
    test_checksum();
    test_result_writer();
    
    // Add more test cases as needed
    printf("All tests passed!\n");
    
    return 0;
}
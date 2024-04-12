#include <assert.h>
#include <vector>
#include <string>

#include "utils/position_result/position_result.h"
#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"

void test_checksum_full_result(){
    // Equal token data & correct token -> equal checksums
    float token_data1[] = {0.1, 0.2, 0.7};
    PositionFullResult result1(
        std::vector<float>(token_data1, token_data1 + 3),
        1
    );
    float token_data2[] = {0.1, 0.2, 0.7};
    PositionFullResult result2(
        std::vector<float>(token_data2, token_data2 + 3),
        1
    );
    assert(result1.getChecksum() == result2.getChecksum());

    // Different correct tokens -> different checksums
    float token_data3[] = {0.1, 0.2, 0.7};
    PositionFullResult result3(
        std::vector<float>(token_data3, token_data3 + 3),
        0
    );
    assert(result1.getChecksum() != result3.getChecksum());

    // Different token data -> different checksums
    float token_data4[] = {0.1, 0.3, 0.6};
    PositionFullResult result4(
        std::vector<float>(token_data4, token_data4 + 3),
        1
    );
    assert(result1.getChecksum() != result4.getChecksum());

    // Different token data & different correct token -> different checksums
    float token_data5[] = {0.1, 0.5, 0.4};
    PositionFullResult result5(
        std::vector<float>(token_data5, token_data5 + 3),
        2
    );
    assert(result1.getChecksum() != result5.getChecksum());

    printf("Checksum test for `PositionFullResult` passed!\n");
}


void test_checksum_top_result(){
    uint16_t k = 2; // top-k

    // Equal token data & correct token -> equal checksums
    float token_data1[] = {0.1, 0.2, 0.7};
    PositionTopResult result1(
        std::vector<float>(token_data1, token_data1 + 3),
        1,
        k
    );
    float token_data2[] = {0.1, 0.2, 0.7};
    PositionTopResult result2(
        std::vector<float>(token_data2, token_data2 + 3),
        1,
        k
    );
    assert(result1.getChecksum() == result2.getChecksum());

    // Different correct tokens -> different checksums
    float token_data3[] = {0.1, 0.2, 0.7};
    PositionTopResult result3(
        std::vector<float>(token_data3, token_data3 + 3),
        0,
        k
    );
    assert(result1.getChecksum() != result3.getChecksum());

    // Different token data -> different checksums
    float token_data4[] = {0.1, 0.3, 0.6};
    PositionTopResult result4(
        std::vector<float>(token_data4, token_data4 + 3),
        1,
        k
    );
    assert(result1.getChecksum() != result4.getChecksum());

    // Different token data & different correct token -> different checksums
    float token_data5[] = {0.1, 0.5, 0.4};
    PositionTopResult result5(
        std::vector<float>(token_data5, token_data5 + 3),
        2,
        k
    );
    assert(result1.getChecksum() != result5.getChecksum());

    printf("Checksum test for `PositionTopResult` passed!\n");
}

int main(){
    test_checksum_full_result();
    test_checksum_top_result();
}
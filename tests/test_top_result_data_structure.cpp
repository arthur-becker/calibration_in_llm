
#include <iostream>
#include <vector>
#include <cassert>

#include "utils/position_result/position_top_result.h"

int main(){  
    uint16_t k = 4; // top-k
    float token_data[] = {5.0, 7.0, 1.0, 6.0, 3.0, 5.0, -1.0, 8.0};
    uint16_t correct_token = 2; // token_data[correct_token] = 1.0

    PositionTopResult result(
        std::vector<float>(token_data, token_data + 8),
        correct_token,
        k
    );

    // Should contain: {1.0 (token_data[correct_token]), 8.0, 7.0, 6.0}
    assert(result.getTokenData().size() == k);
    assert(result.getTokenData()[0] == token_data[correct_token]);
    std::vector<float> top_k = result.getTokenData();
    assert(std::find(top_k.begin(), top_k.end(), 8.0) != top_k.end());
    assert(std::find(top_k.begin(), top_k.end(), 7.0) != top_k.end());
    assert(std::find(top_k.begin(), top_k.end(), 6.0) != top_k.end());
    assert(std::find(top_k.begin(), top_k.end(), 5.0) == top_k.end());
    assert(std::find(top_k.begin(), top_k.end(), 3.0) == top_k.end());
    assert(std::find(top_k.begin(), top_k.end(), -1.0) == top_k.end());
    
    printf("Top result data structure test passed!\n");

    return 0;
}
#include <cassert>
#include <fstream>
#include <algorithm>

#include "utils/position_result/position_top_result.h"

PositionTopResult::PositionTopResult(std::vector<float> token_data, uint16_t correct_token, uint16_t n) : token_data(n, 0.0f) {
    this->n = n;

    assert(n <= token_data.size());
    
    // Save the correct token data
    float correct_token_data = token_data[correct_token];
    this->token_data[0] = correct_token_data;

    // Save the k-1 other tokens with the highest probability
    float min_token_data = *std::min_element(token_data.begin(), token_data.end());
    token_data[correct_token] = min_token_data;
    std::sort(token_data.begin(), token_data.end(), std::greater<float>());
    for (uint16_t i = 0; i < n - 1; i++) {
        this->token_data[i + 1] = token_data[i];
    }
}

PositionTopResult PositionTopResult::fromFile(std::ifstream * file){
    // Assumption for consistency of checksum calculation
    static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)"); 

    // 1. Read number of tokens
    uint8_t float_size = sizeof(float);
    uint16_t n;
    file->read(reinterpret_cast<char*>(&n), sizeof(n));

    if(file->eof()){
        throw EOFException();
    }

    // 2. Read token data
    std::vector<float> token_data(n, 0.0f);
    for (int i = 0; i < n; i++) {
        float token;
        file->read(reinterpret_cast<char*>(&token), float_size);
        token_data[i] = token;
    }

    return PositionTopResult(token_data, 0, n);
}

std::vector<float> PositionTopResult::getTokenData() const {
    return this->token_data;
}

std::vector<uint8_t> PositionTopResult::getBytes() const {
    // Assumption for consistency of checksum calculation
    static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)"); 

    uint8_t float_size = sizeof(float);
    std::vector<uint8_t> data(
        countBytes()); // save the number of tokens and the token data
    auto start_n_bytes = data.begin();
    auto start_token_data = start_n_bytes + sizeof(this->n);

    // 1. Save number of tokens
    const uint8_t * n_bytes = reinterpret_cast<const uint8_t*>(&this->n);
    std::copy(n_bytes, n_bytes + sizeof(this->n), start_n_bytes);

    // 2. Save token data
    for (float token : this->token_data) {
        uint8_t * token_bytes = reinterpret_cast<uint8_t*>(&token);
        std::copy(token_bytes, token_bytes + float_size, start_token_data);
        start_token_data += float_size;
    }

    return data;
}

uint16_t PositionTopResult::getN() const {
    return this->n;
}

uint32_t PositionTopResult::countBytes() const {
    // Assumption for consistency of checksum calculation
    static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)"); 

    uint8_t float_size = sizeof(float);
    uint16_t bytes_number = sizeof(this->n) + this->n * float_size;

    return bytes_number;
}

#include <fstream>
#include <utility>
#include <algorithm>

#include "utils/position_result/position_full_result.h"

PositionFullResult::PositionFullResult(std::vector<float> token_data, uint16_t correct_token) : token_data(std::move(token_data)), correct_token(correct_token) {}

PositionFullResult PositionFullResult::fromFile(std::ifstream * file){
    // Assumption for consistency of checksum calculation
    static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)");

    // 1. Read number of tokens
    uint8_t float_size = sizeof(float);
    uint16_t n;
    file->read(reinterpret_cast<char*>(&n), sizeof(n));

    if(file->eof()){
        throw EOFException();
    }

    // 2. Read correct token
    uint16_t correct_token;
    file->read(reinterpret_cast<char*>(&correct_token), sizeof(correct_token));

    // 3. Read token data
    std::vector<float> token_data(n, 0.0f);
    for (int i = 0; i < n; i++) {
        float token;
        file->read(reinterpret_cast<char*>(&token), float_size);
        token_data[i] = token;
    }

    return PositionFullResult(token_data, correct_token);
}

std::vector<float> PositionFullResult::getTokenData() const {
    return this->token_data;
}

std::vector<uint8_t> PositionFullResult::getBytes() const {
    // Assumption for consistency of checksum calculation
    static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)");

    // Check if this->token_data.size() fits into uint16_t
    if (this->token_data.size() > UINT16_MAX) {
        throw std::runtime_error("Token data size exceeds uint16_t");
    }
    uint16_t n = this->token_data.size(); // vocabulary size

    std::vector<uint8_t> data(this->countBytes()); // save the number of tokens, the correct token and the token data
    auto start_n_bytes = data.begin();
    auto start_correct_token = start_n_bytes + sizeof(correct_token);
    auto start_token_data = start_correct_token + sizeof(correct_token);

    // 1. Save number of tokens
    uint8_t * n_bytes = reinterpret_cast<uint8_t*>(&n);
    std::copy(n_bytes, n_bytes + sizeof(n), start_n_bytes);

    // 2. Save correct token
    const uint8_t * correct_token_bytes = reinterpret_cast<const uint8_t*>(&this->correct_token);
    std::copy(correct_token_bytes, correct_token_bytes + sizeof(this->correct_token), start_correct_token);

    // 3. Save token data
    std::copy_n(reinterpret_cast<const uint8_t*>(token_data.data()),
                token_data.size() * sizeof(float),
                start_token_data);

    return data;
}

uint16_t PositionFullResult::getCorrectToken() const {
    return this->correct_token;
}

std::size_t PositionFullResult::countBytes() const {
    // Assumption for consistency of checksum calculation
    static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)");

    std::size_t float_size = sizeof(float);
    std::uint16_t n = this->token_data.size(); // vocabulary size
    std::size_t bytes_number = sizeof(n) + sizeof(this->correct_token) + n * float_size;

    return bytes_number;
}

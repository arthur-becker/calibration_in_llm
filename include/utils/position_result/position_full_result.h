#include <fstream>
#include <vector>

#include "utils/position_result/position_result.h"

#pragma once

/// @brief This class represents the result of a single position in the input sequence.
/// The positional information is preserved in the vector returned by `getTokenData()`, e.g. the vector contains
/// data for each token in the vocabulary, and the element at index i corresponds to the token=i
class PositionFullResult : public PositionResult {
private:
    const std::vector<float> token_data;
    uint16_t correct_token;
public:
    PositionFullResult(std::vector<float> token_data, uint16_t correct_token);
    static PositionFullResult fromFile(std::ifstream * file);

    std::vector<float> getTokenData() const override;
    std::vector<uint8_t> getBytes() const override;
    uint32_t countBytes() const override;
    uint16_t getCorrectToken() const;
};
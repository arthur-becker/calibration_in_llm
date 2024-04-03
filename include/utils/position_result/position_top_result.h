#include <fstream>
#include <vector>

#include "utils/position_result/position_result.h"

#pragma once

/// @brief This class represents the result of a single position in the input sequence. In contrast to `FullPositionResult`,
/// this class does not preserve the positional information about the token, and only stores the results for the top-k or top-p tokens.
///
/// The only important positional information is the correct token, which is stored as the first element in the `token_data` vector.
class PositionTopResult : public PositionResult {
private:
    std::vector<float> token_data;
    uint16_t n;
public:
    PositionTopResult(std::vector<float> token_data, uint16_t correct_token, uint16_t n);
    static PositionTopResult fromFile(std::ifstream * file);

    std::vector<float> getTokenData() const override;
    std::vector<uint8_t> getBytes() const override;
    std::size_t countBytes() const override;
    uint16_t getN() const;
};
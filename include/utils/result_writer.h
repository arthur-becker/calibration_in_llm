#include <vector>
#include <string>
#include <fstream>

#pragma once

class PositionResult {
private:
    const std::vector<float> token_data;
    const int8_t correct_token;
public:
    PositionResult(std::vector<float> token_data, uint16_t correct_token);
    std::vector<float> getTokenData();
    uint16_t getCorrectToken();
    uint32_t getChecksum();
};

class ResultWriter {
private:
    std::string filename;
    std::vector<PositionResult> result;
    std::ofstream file;

public:
    ResultWriter(std::string filename);
    void openFile();
    void closeFile();
    void addPositionResult(PositionResult position_result);
    void writeAndClear();
};
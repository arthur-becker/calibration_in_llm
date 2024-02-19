#include <fstream>
#include <vector>
#include <string>

#include "utils/position_result/position_result.h"

#pragma once

template <typename T>
class ResultWriter {
    static_assert(std::is_base_of<PositionResult, T>::value, "T must be a derived class of PositionResult");
private:
    std::string filename;
    std::vector<T> result;
    std::ofstream file;

public:
    ResultWriter(std::string filename);
    void openFile();
    void closeFile();
    void addPositionResult(T position_result);
    void writeAndClear();
};
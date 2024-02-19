#pragma once

#include <vector>
#include "utils/result_writer.h"

template <typename T>
class ResultReader {
    static_assert(std::is_base_of<PositionResult, T>::value, "T must be a derived class of PositionResult");

public:
    std::vector<T> read_result(std::string file_name);
};
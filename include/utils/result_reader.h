#pragma once

#include <vector>
#include "result_writer.h"

class ResultReader {
public:
    std::vector<PositionResult> read_result(std::string file_name);
};
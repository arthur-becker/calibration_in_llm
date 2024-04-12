#include <zlib.h>
#include <cstring>
#include <utility>
#include <vector>
#include <fstream>

#include "utils/result_writer.h"
#include "utils/position_result/position_result.h"
#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"


template<typename T>
ResultWriter<T>::ResultWriter(std::string filename) : filename(std::move(filename)) {}

template<typename T>
void ResultWriter<T>::openFile() {
    this->file.open(this->filename);

    if (!this->file.is_open()) {
        printf("Failed to open file %s\n", this->filename.c_str());
        exit(1);
    }
}

template<typename T>
void ResultWriter<T>::closeFile() {
    this->file.close();
}

template<typename T>
void ResultWriter<T>::addPositionResult(T position_result) {
    this->result.push_back(std::move(position_result));
}

template<typename T>
void ResultWriter<T>::writeAndClear() {
    for (const T& position_result : this->result) {
        const std::vector<uint8_t> data = position_result.getBytes();
        this->file.write(reinterpret_cast<const char*>(data.data()), data.size());

        // Write checksum
        const uint32_t checksum = position_result.getChecksum();
        this->file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    }

    this->result.clear();
}

template class ResultWriter<PositionFullResult>;

template class ResultWriter<PositionTopResult>;
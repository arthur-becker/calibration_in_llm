#include "utils/result_writer.h"
#include <zlib.h>
#include <assert.h>

/*
PositionResult class
*/
PositionResult::PositionResult(std::vector<float> token_data, uint16_t correct_token) : token_data(token_data), correct_token(correct_token) {}

std::vector<float> PositionResult::getTokenData() {
    return this->token_data;
}

uint16_t PositionResult::getCorrectToken() {
    return this->correct_token;
}

uint32_t PositionResult::getChecksum() {
    // Assumption for consistency of checksum calculation
    static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)"); 

    uint8_t float_size = sizeof(float);

    std::vector<uint8_t> data(sizeof(this->correct_token) + this->token_data.size() * float_size);
    uint8_t* data_ptr = data.data();

    // Copy correct token
    memcpy(data_ptr, &this->correct_token, sizeof(this->correct_token));
    data_ptr += sizeof(this->correct_token);

    // Copy token data
    for (float token : this->token_data) {
        memcpy(data_ptr, &token, float_size);
        data_ptr += float_size;
    }

    // Calculate checksum
    uint32_t checksum = crc32(0L, Z_NULL, 0);
    checksum = crc32(checksum, data.data(), data.size());

    return checksum;
}

/*
OutputWriter class
*/
ResultWriter::ResultWriter(std::string filename) : filename(filename) {}

void ResultWriter::openFile() {
    this->file.open(this->filename);

    if (!this->file.is_open()) {
        printf("Failed to open file %s\n", this->filename.c_str());
        exit(1);
    }
}

void ResultWriter::closeFile() {
    this->file.close();
}

void ResultWriter::addPositionResult(PositionResult position_result) {
    this->result.push_back(position_result);
}

void ResultWriter::writeAndClear() {
    printf("[ResultWriter] Writing %zu results to file %s\n", this->result.size(), this->filename.c_str());
    for (PositionResult position_result : this->result) {
        // Write correct token
        uint16_t correct_token = position_result.getCorrectToken();
        this->file.write(reinterpret_cast<char*>(&correct_token), sizeof(correct_token));

        // Write number of saved tokens
        uint16_t num_saved_tokens = position_result.getTokenData().size();
        this->file.write(reinterpret_cast<char*>(&num_saved_tokens), sizeof(num_saved_tokens));

        // Write token data
        for (float token : position_result.getTokenData()) {
            this->file.write(reinterpret_cast<char*>(&token), sizeof(token));
        }

        // Write checksum
        uint32_t checksum = position_result.getChecksum();
        this->file.write(reinterpret_cast<char*>(&checksum), sizeof(checksum));
    }

    this->result.clear();
}
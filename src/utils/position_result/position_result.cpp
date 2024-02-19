#include <zlib.h>

#include "utils/position_result/position_result.h"

uint32_t PositionResult::getChecksum() const {
    std::vector<uint8_t> data = this->getBytes();

    // Calculate checksum
    uint32_t checksum = crc32(0L, Z_NULL, 0);
    checksum = crc32(checksum, data.data(), data.size());

    return checksum;
}

const char * EOFException::what() const noexcept {
    return "End of file reached\n";
}
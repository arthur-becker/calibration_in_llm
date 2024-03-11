#include <vector>
#include <cstdint>

#pragma once

class PositionResult {
public:
    virtual std::vector<float> getTokenData() const = 0;
    virtual std::vector<std::uint8_t> getBytes() const = 0;

    // @brief Returns the number of bytes that the PositionResult object occupies in memory when serialized with `getBytes()`
    virtual std::uint32_t countBytes() const = 0;
    std::uint32_t getChecksum() const;
    virtual ~PositionResult() = default;
};

class EOFException : public std::exception {
public:
    const char * what() const noexcept override;
};
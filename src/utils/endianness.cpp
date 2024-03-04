#include <iostream>

bool isLittleEndian(){
    uint16_t word = 1; // 0x0001
    uint8_t *first_byte = (uint8_t*) &word; // points to the first byte of word
    return *first_byte; // true if the first byte is zero
}
# Calibration in quantized LLMs

## How to build
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## How to run tests
```bash
cd build/tests
ctest -V
```

## Results file format
> Assumption: `sizeof(float) == sizeof(uint_32) == 4 bytes`

For every `PositionResult`:
    1. `uint16_t correct_token` (2 bytes) 
    2. `uint16_t num_saved_tokens` (2 bytes)
    3. For `num_saved_tokens` times: 
        - `float token_data` (4 bytes)
    4. `uint32_t checksum` (4 bytes).
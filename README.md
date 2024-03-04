# Calibration in quantized LLMs

## How to build
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## How to run evaluation script

First, navigate to the scripts folder:
```bash
cd scripts
```

Install dependencies:
```bash
pip3 install -r requirements.txt
``` 

Then, run the evaluation script
```bash
python3 evaluate.py <output_folder_name>
```

where `output_folder_name` should be the folder name in `results/` where the script `extract_probabilities.cpp` saves logits and probabilities



## How to run tests
For C++ part:
```bash
cd build/tests
ctest -V
```

For Python:
```bash
cd scripts
python -m unittest tests/result_reader_test.py
```

## Results file format
> Assumption: `sizeof(float) == sizeof(uint_32) == 4 bytes`

For every `PositionResult`:
    1. `uint16_t correct_token` (2 bytes) 
    2. `uint16_t num_saved_tokens` (2 bytes)
    3. For `num_saved_tokens` times: 
        - `float token_data` (4 bytes)
    4. `uint32_t checksum` (4 bytes).
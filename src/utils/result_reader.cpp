#include "utils/result_reader.h"
#include "utils/result_writer.h"

#include <vector>
#include <string>
#include <fstream>

void print_result(PositionResult result) {
    printf("\nPosition result:\n");
    printf("Correct token: %d\n", result.getCorrectToken());
    printf("Token data: ");
    for (float token : result.getTokenData()) {
        printf("%f ", token);
    }
    printf("\n");
    printf("Checksum: %d\n\n", result.getChecksum());
}

std::vector<PositionResult> ResultReader::read_result(std::string file_name) {
    std::vector<PositionResult> result;
    std::ifstream file(file_name, std::ios::binary);

    if(sizeof(float) != sizeof(uint32_t)){
        fprintf(stderr, "Error: float size is not equal to uint32_t size\n");
        exit(1);
    }

    if (!file.is_open()) {
        fprintf(stderr, "Error: unable to open file %s\n", file_name.c_str());
        exit(1);
    }


    while(true){
        // Read correct token
        uint16_t correct_token;
        file.read(reinterpret_cast<char*>(&correct_token), sizeof(correct_token));
        printf("[ResultReader] correct_token: %d\n", correct_token);

        // Check if EOF
        if(file.eof()){
            break;
        }

        // Read number of saved tokens
        uint16_t num_saved_tokens;
        file.read(reinterpret_cast<char*>(&num_saved_tokens), sizeof(num_saved_tokens));
        printf("[ResultReader] num_saved_tokens: %d\n", num_saved_tokens);

        // Read token data
        std::vector<float> token_data;
        for (int i = 0; i < num_saved_tokens; i++) {
            float token;
            file.read(reinterpret_cast<char*>(&token), sizeof(float));
            token_data.push_back(token);
        }
        
        // Create PositionResult object
        PositionResult position_result(token_data, correct_token);

        // Read checksum and compare it with the calculated checksum
        uint32_t checksum;
        file.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
        if (checksum != position_result.getChecksum()) {
            fprintf(stderr, "Error: checksum mismatch. Print result:\n");
            print_result(position_result);
            exit(1);
        }

        result.push_back(position_result);  
    }

    return result;
};


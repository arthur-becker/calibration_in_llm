#include "utils/result_reader.h"
#include "utils/result_writer.h"
#include "utils/position_result/position_result.h"
#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"

#include <vector>
#include <string>
#include <fstream>

void print_result(PositionFullResult result) {
    printf("\nPosition result:\n");
    printf("Correct token: %d\n", result.getCorrectToken());
    printf("Token data: ");
    for (float token : result.getTokenData()) {
        printf("%f ", token);
    }
    printf("\n");
    printf("Checksum: %d\n\n", result.getChecksum());
}

template <typename T>
std::vector<T> ResultReader<T>::read_result(std::string file_name) {
    std::vector<T> result;
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
        try {
            T position_result = T::fromFile(&file);

            uint32_t checksum;
            file.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
            if (checksum != position_result.getChecksum()) {
                throw std::runtime_error("Checksum mismatch");
            }

            result.push_back(position_result);
        } catch (EOFException &e) {
            break;
        } catch (std::exception &e) {
            fprintf(stderr, "Error: %s\n", e.what());
            exit(1);
        }
    }

    return result;
};

template class ResultReader<PositionFullResult>;
template class ResultReader<PositionTopResult>;


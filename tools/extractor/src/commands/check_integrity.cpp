#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "utils/result_reader.h"
#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <file_path> <output_writer_type>\n", argv[0]);
        exit(1);
    }
    std::string file_path = std::move(argv[1]);
    std::string output_writer_type = std::move(argv[2]);

    if (output_writer_type == "full") {
        printf("Output writer type: full\n");
        
        ResultReader<PositionFullResult> reader;
        try
        {
            std::vector<PositionFullResult> read_results = reader.read_result(file_path);
            printf("Successfully read from file: %s\n", file_path.c_str());
        }
        catch(const std::exception& e)
        {
            fprintf(stderr, "Error reading file: %s\n", e.what());
            exit(1);
        }

    } else if (output_writer_type == "top-k") {
        printf("Output writer type: top-k\n");

        ResultReader<PositionTopResult> reader;
        try
        {
            std::vector<PositionTopResult> read_results = reader.read_result(file_path);
            printf("Successfully read from file: %s\n", file_path.c_str());
        }
        catch(const std::exception& e)
        {
            fprintf(stderr, "Error reading file: %s\n", e.what());
            exit(1);
        }
    } else {
        fprintf(stderr, "Unknown output writer type: %s\n", output_writer_type.c_str());
        exit(1);
    }

    return 0;
}
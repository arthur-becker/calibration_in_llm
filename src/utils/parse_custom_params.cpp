#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>

#include "utils/parse_custom_params.h"

std::map<std::string, std::string> args_to_map(int * argc, char *** argv, char * custom_params_names[]){
    std::map<std::string, std::string> custom_params_map;
    for (int i = 1; i < *argc; i++) {
        for (int j = 0; custom_params_names[j] != NULL; j++) {
            if (std::strcmp((*argv)[i], custom_params_names[j]) == 0) {
                if (i + 1 >= *argc) {
                    fprintf(stderr, "Error: %s requires an argument\n", custom_params_names[j]);
                    exit(1);
                }

                custom_params_map[custom_params_names[j]] = (*argv)[i + 1];

                for (int k = i; k < *argc - 2; k++) {
                    (*argv)[k] = (*argv)[k + 2];
                }
                *argc -= 2;
                i--;
            }
        }
    }
    return custom_params_map;

}

CustomParams parse_custom_params(int * argc, char *** argv){
    char * custom_params_names[] = {
        (char *) "--output-writer-type",
        (char *) "--write-top-k",
        NULL
    };
    auto custom_params_map = args_to_map(argc, argv, custom_params_names);

    CustomParams custom_params;

    // Default values
    custom_params.output_writer_type = OutputWriterType::TOP_K;
    custom_params.top_k = 100;

    if (custom_params_map.find("--output-writer-type") != custom_params_map.end()) {
        if (custom_params_map["--output-writer-type"] == "full") {
            custom_params.output_writer_type = OutputWriterType::FULL;
        } else if (custom_params_map["--output-writer-type"] == "top-k") {
            custom_params.output_writer_type = OutputWriterType::TOP_K;
            if (custom_params_map.find("--write-top-k") == custom_params_map.end())
            {
                printf("--output-writer-type is not specified, using default value of 100 for --write-top-k\n");
            }
            
        } else {
            fprintf(stderr, "Error: --output-writer-type must be either 'full' or 'top-k'\n");
            exit(1);
        }
    }

    if (custom_params_map.find("--write-top-k") != custom_params_map.end()) {
        custom_params.top_k = std::atoi(custom_params_map["--write-top-k"].c_str());
    }

    return custom_params;
}
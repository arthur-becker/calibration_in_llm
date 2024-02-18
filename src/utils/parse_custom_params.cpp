#include <stdio.h>
#include <cstring>
#include <cstdlib>

#include "utils/parse_custom_params.h"

CustomParams parse_custom_params(int * argc, char *** argv){
    CustomParams custom_params;
    custom_params.output_writer_type = OutputWriterType::FULL;

    for (int i = 1; i < *argc; i++) {
        if (std::strcmp((*argv)[i], "--output-writer-type") == 0) {
            if (i + 1 >= *argc) {
                fprintf(stderr, "Error: --output-writer-type requires an argument\n");
                exit(1);
            }
            if (strcmp((*argv)[i + 1], "full") == 0) {
                custom_params.output_writer_type = OutputWriterType::FULL;
            } else if (strcmp((*argv)[i + 1], "top") == 0) {
                custom_params.output_writer_type = OutputWriterType::TOP;
            } else {
                fprintf(stderr, "Error: --output-writer-type requires an argument of either 'full' or 'top'\n");
                exit(1);
            }
            for (int j = i; j < *argc - 2; j++) {
                (*argv)[j] = (*argv)[j + 2];
            }
            *argc -= 2;
            i--;
        }
    }

    return custom_params;
}
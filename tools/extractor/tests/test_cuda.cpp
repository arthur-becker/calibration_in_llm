#include "common.h"

/// @brief It has been observed that the gpt_params_parse function causes 
/// a segmentation fault when the project is compiled with CUDA support, and works fine without it.
///
/// This is a minimal example to reproduce the issue. 
int main(){  
    gpt_params params;
    int argc = 1;
    char ** argv = new char*[1];
    argv[0] = new char[1];
    argv[0][0] = '\0';

    gpt_params_parse(argc, argv, params);

    return 0;
}
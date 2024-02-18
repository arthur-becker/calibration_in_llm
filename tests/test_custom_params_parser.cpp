#include <iostream>
#include <cassert>

#include "utils/parse_custom_params.h"

bool compare_arrays(int argc, char ** argv, int argc2, char ** argv2){
    if (argc != argc2) return false;
    for (int i = 0; i < argc; i++){
        if (std::strcmp(argv[i], argv2[i]) != 0) return false;
    }
    return true;
}

int main(){
    int argc1 = 7;
    char ** argv1 = new char*[argc1];
    argv1[0] = (char*)"./extract_probabilities";
    argv1[1] = (char*)"-m";
    argv1[2] = (char*)"/path/to/model";
    argv1[3] = (char*)"--output-writer-type";
    argv1[4] = (char*)"top";
    argv1[5] = (char*)"-f";
    argv1[6] = (char*)"file.txt";

    int argc2 = 5;
    char ** argv2 = new char*[argc2];
    argv2[0] = (char*)"./extract_probabilities";
    argv2[1] = (char*)"-m";
    argv2[2] = (char*)"/path/to/model";
    argv2[3] = (char*)"-f";
    argv2[4] = (char*)"file.txt";


    assert(!compare_arrays(argc1, argv1, argc2, argv2));
    CustomParams custom_params = parse_custom_params(&argc1, &argv1);
    assert(custom_params.output_writer_type == OutputWriterType::TOP);
    assert(compare_arrays(argc1, argv1, argc2, argv2));

    return 0;
}
#include <iostream>
#include <cassert>
#include <cstring>
#include <map>

#include "utils/parse_custom_params.h"

bool compare_arrays(int argc, char ** argv, int argc2, char ** argv2){
    if (argc != argc2) return false;
    for (int i = 0; i < argc; i++){
        if (std::strcmp(argv[i], argv2[i]) != 0) return false;
    }
    return true;
}

void test_args_to_map(){
    int argc = 7;
    char ** argv = new char*[argc];
    argv[0] = (char*)"./extract_probabilities";
    argv[1] = (char*)"-m";
    argv[2] = (char*)"/path/to/model";
    argv[3] = (char*)"--output-writer-type";
    argv[4] = (char*)"top-k";
    argv[5] = (char*)"-f";
    argv[6] = (char*)"file.txt";

    char * custom_params_names[] = {
        (char *) "--output-writer-type",
        (char *) "--write-top-k",
        NULL
    };
    std::map<std::string, std::string> custom_params_map = args_to_map(&argc, &argv, custom_params_names);
    assert(custom_params_map.size() == 1);
    assert(custom_params_map["--output-writer-type"] == "top-k");
    assert(compare_arrays(argc, argv, 5, new char*[5]{
        (char*)"./extract_probabilities",
        (char*)"-m",
        (char*)"/path/to/model",
        (char*)"-f",
        (char*)"file.txt"
    }));
    printf("args_to_map passed\n");
}

void test_parse_custom_params(){
    int argc = 11;
    char ** argv = new char*[argc];
    argv[0] = (char*)"./extract_probabilities";
    argv[1] = (char*)"-m";
    argv[2] = (char*)"/path/to/model";
    argv[3] = (char*)"--output-writer-type";
    argv[4] = (char*)"top-k";
    argv[5] = (char*)"-f";
    argv[6] = (char*)"file.txt";
    argv[7] = (char*)"--write-top-k";
    argv[8] = (char*)"50";
    argv[9] = (char*)"--output-folder";
    argv[10] = (char*)"/path/to/output/folder";

    CustomParams custom_params = parse_custom_params(&argc, &argv);
    assert(custom_params.output_writer_type == OutputWriterType::TOP_K);
    assert(custom_params.top_k == 50);
    assert(custom_params.output_folder == (char*)"/path/to/output/folder");
    assert(compare_arrays(argc, argv, 5, new char*[5]{
        (char*)"./extract_probabilities",
        (char*)"-m",
        (char*)"/path/to/model",
        (char*)"-f",
        (char*)"file.txt"
    }));
    printf("parse_custom_params passed\n");
}

int main(){
    test_args_to_map();
    test_parse_custom_params();
    return 0;
}
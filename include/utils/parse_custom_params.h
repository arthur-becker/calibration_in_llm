#pragma once

enum OutputWriterType {
    FULL,
    TOP_K
};

struct CustomParams{
    OutputWriterType output_writer_type;
    uint16_t top_k;
};

/// @brief This function converts the custom parameters from the command line arguments to a map.
/// @param argc  Number of command line arguments
/// @param argv  Command line arguments
/// @param custom_params_names  Names of the custom parameters. This array should be NULL-terminated.
/// @return   Map of custom parameters
std::map<std::string, std::string> args_to_map(int * argc, char *** argv, char * custom_params_names[]);

/// @brief This function parses custom parameters from the command line arguments. The custom parameters are
///        the ones that are not parsed by the standard llama.cpp parser. The function removes the parsed
///        custom parameters from `argv` and updates `argc` accordingly in order to prevent any errors
///        when parsing the standard parameters with `gpt_params_parse`.
/// @param argc  Number of command line arguments
/// @param argv  Command line arguments
/// @return   Parsed custom parameters
CustomParams parse_custom_params(int * argc, char *** argv);
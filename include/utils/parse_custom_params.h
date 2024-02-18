enum OutputWriterType {
    FULL,
    TOP
};

struct CustomParams{
    OutputWriterType output_writer_type;
};

/// @brief This function parses custom parameters from the command line arguments. The custom parameters are
///        the ones that are not parsed by the standard llama.cpp parser. The function removes the parsed
///        custom parameters from `argv` and updates `argc` accordingly in order to prevent any errors
///        when parsing the standard parameters with `gpt_params_parse`.
/// @param argc  Number of command line arguments
/// @param argv  Command line arguments
/// @return   Parsed custom parameters
CustomParams parse_custom_params(int * argc, char *** argv);
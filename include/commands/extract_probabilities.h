

#include <vector>

#include "llama.h"
#include "utils/parse_custom_params.h"
#include "utils/llama_cpp_helper.h"
#include "utils/input_iterator.h"
#include "utils/result_writer.h"
#include "utils/position_result/position_result.h"

#pragma once

template <typename T>
class ProbabilitiesExtractor : public LlamaCppHelper {
    static_assert(std::is_base_of<PositionResult, T>::value, "T must be a subclass of PositionResult");

private:
    CustomParams custom_params;

    InputIterator<llama_token> * input_iterator;
    std::vector<llama_token> tokens;
    ResultWriter<T> logits_writer = {""};
    ResultWriter<T> proba_writer = {""};

    std::vector<float> get_chunk_logits(Chunk chunk);
    void write_chunk_logits_and_proba(std::vector<float> * logits, Chunk chunk);
    void init_result_writers(std::string output_folder, OutputWriterType output_writer_type);
    void save_run_info(gpt_params params, CustomParams custom_params);
    std::string getOutputFolderPath(std::string output_folder);

public:
    ProbabilitiesExtractor(int argc, char ** argv, CustomParams custom_params);

    void tokenize(std::string &prompt);
    void run();
};
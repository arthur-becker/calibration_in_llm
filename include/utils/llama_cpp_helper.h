#include "llama.h"
#include "common.h"

#pragma once

class LlamaCppHelper {
private:
    llama_model * model;
    llama_context * ctx;

    gpt_params params;

    int n_ctx; // Context size
    int n_batch; // Batch size
    int n_vocab; // Vocab size
    bool add_bos;

    /// @brief Parse standard parameters of llama.cpp from command line and set defaults
    /// @param argc  Number of command line arguments
    /// @param argv  Command line arguments
    void init_llama_cpp_params(int argc, char ** argv);

    void load_model();

    void free();
public:
    LlamaCppHelper(int argc, char ** argv);
    ~LlamaCppHelper();

    llama_context * getContext();
    llama_model * getModel();
    gpt_params getParams();
    int getContextSize() const;
    int getBatchSize() const;
    int getVocabSize() const;
    bool shouldAddBOS() const;

    llama_token getTokenBOS();
};
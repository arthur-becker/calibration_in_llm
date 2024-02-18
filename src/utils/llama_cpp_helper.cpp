#include "utils/llama_cpp_helper.h"
#include "common.h"

LlamaCppHelper::LlamaCppHelper(int argc, char ** argv){
    this->init_llama_cpp_params(argc, argv);
    this->load_model();

    // Parameters
    this->n_ctx = llama_n_ctx(ctx); // Context size
    this->n_batch = params.n_batch; // Batch size
    this->n_vocab = llama_n_vocab(llama_get_model(ctx)); // Vocab size
    this->add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    printf("n_ctx: %d\n", this->n_ctx);
    printf("n_batch: %d\n", this->n_batch);
    printf("n_vocab: %d\n", this->n_vocab);
    printf("add_bos: %d\n", this->add_bos);
}

LlamaCppHelper::~LlamaCppHelper(){
    this->free();
}

llama_context * LlamaCppHelper::getContext(){
    return this->ctx;
}

llama_model * LlamaCppHelper::getModel(){
    return this->model;
}

gpt_params LlamaCppHelper::getParams(){
    return this->params;
}

int LlamaCppHelper::getContextSize(){
    return this->n_ctx;
}

int LlamaCppHelper::getBatchSize(){
    return this->n_batch;
}

int LlamaCppHelper::getVocabSize(){
    return this->n_vocab;
}

bool LlamaCppHelper::shouldAddBOS(){
    return this->add_bos;
}

llama_token LlamaCppHelper::getTokenBOS(){
    return llama_token_bos(llama_get_model(this->ctx));
}

void LlamaCppHelper::init_llama_cpp_params(int argc, char ** argv){
    this->params.n_batch = 512;
    if (!gpt_params_parse(argc, argv, params)) {
        printf("Failed to parse command line arguments\n");
        exit(1);
    }

    this->params.logits_all = true;
    this->params.n_batch = std::min(params.n_batch, params.n_ctx);

    /*if (this->params.ppl_stride > 0) {
        fprintf(stderr, "Will perform strided perplexity calculation -> adjusting context size from %d to %d\n",
                this->params.n_ctx, this->params.n_ctx + this->params.ppl_stride/2);
        this->params.n_ctx += params.ppl_stride/2;
    }*/

    if (this->params.seed == LLAMA_DEFAULT_SEED) {
        this->params.seed = time(NULL);
    }

    std::mt19937 rng(this->params.seed);
    if (this->params.random_prompt) {
        this->params.prompt = gpt_random_prompt(rng);
    }
    fprintf(stderr, "%s: seed  = %u\n", __func__, this->params.seed);
}

void LlamaCppHelper::load_model(){
    llama_backend_init(this->params.numa);

    std::tie(this->model, this->ctx) = llama_init_from_gpt_params(this->params);
    if (this->model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        exit(1);
    }

    const int n_ctx_train = llama_n_ctx_train(this->model);
    if (params.n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

}

void LlamaCppHelper::free(){
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}
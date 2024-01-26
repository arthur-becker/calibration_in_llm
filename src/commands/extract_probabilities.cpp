#include "common.h"

#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

gpt_params init_params(int argc, char ** argv){
    gpt_params params;

    params.n_batch = 512;
    if (!gpt_params_parse(argc, argv, params)) {
        printf("Failed to parse command line arguments\n");
        exit(1);
    }

    params.logits_all = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.ppl_stride > 0) {
        fprintf(stderr, "Will perform strided perplexity calculation -> adjusting context size from %d to %d\n",
                params.n_ctx, params.n_ctx + params.ppl_stride/2);
        params.n_ctx += params.ppl_stride/2;
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }
    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    return params;
}

std::tuple<struct llama_model *, struct llama_context *> load_model(gpt_params params){
    llama_model * model;
    llama_context * ctx;

    llama_backend_init(params.numa);

    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        exit(1);
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    return std::make_tuple(model, ctx);
}

void model_free(llama_context * ctx, llama_model * model){
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    gpt_params params = init_params(argc, argv);

    llama_model * model;
    llama_context * ctx;
    std::tie(model, ctx) = load_model(params);
    
    // TODO: implement probability extraction here

    model_free(ctx, model);

    return 0;
}
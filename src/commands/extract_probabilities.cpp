#include "common.h"
#include "utils/input_iterator.h"
#include "utils/result_writer.h"
#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"
#include "utils/position_result/position_result.h"
#include "utils/parse_custom_params.h"

#include <cstdio>
#include <fstream>
#include <cassert>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


template<typename T>
void assert_output_writer_type(CustomParams custom_params, OutputWriterType expected_output_writer_type){
    static_assert(std::is_base_of<PositionResult, T>::value, "T must be a subclass of PositionResult");

    // OutputWriterType::FULL -> PositionFullResult
    assert((custom_params.output_writer_type != OutputWriterType::FULL || std::is_base_of<PositionFullResult, T>::value)); 
    
    // OutputWriterType::TOP_K -> PositionTopResult
    assert((custom_params.output_writer_type != OutputWriterType::TOP_K || std::is_base_of<PositionTopResult, T>::value));

}

/// @brief Parse standard parameters of llama.cpp from command line and set defaults
/// @param argc  Number of command line arguments
/// @param argv  Command line arguments
/// @return     Parsed parameters
gpt_params init_llama_cpp_params(int argc, char ** argv){
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

std::vector<float> get_chunk_logits(
    Chunk chunk,
    InputIterator<llama_token> * input_iterator,
    llama_context * ctx,
    const int n_batch,
    const int n_vocab,
    const bool add_bos
    ){
    std::vector<int> tokens = *(input_iterator->getInput());
    std::vector<float> chunk_logits(chunk.getSize() * n_vocab);

    BatchCallback batch_callback = [&](Batch batch){
        // save original token and restore it after eval
        const auto token_org = tokens[batch.getStart()];

        // add BOS token for the first batch of each chunk
        if (add_bos && batch.getIndex() == 0) {
            tokens[batch.getStart()] = llama_token_bos(llama_get_model(ctx));
        }

        llama_token* tokens_batch = tokens.data() + batch.getStart(); // Pointer to the first token in the batch in the tokens vector
        int32_t batch_size = batch.getSize();
        llama_pos pos_0 = batch.getIndex() * n_batch; // Position in the chunk

        llama_batch batch_data = llama_batch_get_one(tokens_batch, batch_size, pos_0, 0);
        if (llama_decode(ctx, batch_data)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            exit(1);
        }

        // restore the original token in case it was set to BOS
        tokens[batch.getStart()] = token_org;

        const float * batch_logits = llama_get_logits(ctx);
        float * first = (float *) batch_logits; // Pointer to the first logit in the batch
        float * last = first + batch.getSize() * n_vocab; // Pointer to the first logit of the next batch
        chunk_logits.insert(chunk_logits.end(), first, last);
    };
    input_iterator->iterate(batch_callback, chunk);

    return chunk_logits;
}

template<typename T>
void write_chunk_logits(
    std::vector<float> * logits,
    Chunk chunk,
    std::vector<int> * tokens,
    ResultWriter<T> * result_writer,
    const int n_ctx,
    const int n_vocab,
    CustomParams custom_params
    ){
    assert_output_writer_type<T>(custom_params, custom_params.output_writer_type);

    int second_half_start = n_ctx / 2;
    for(int i = second_half_start; i < n_ctx; i++){
        float * first = (float *) logits->data() + i * n_vocab; 
        float * last = first + n_vocab; 
        std::vector<float> token_data(first, last);

        int token_positon = chunk.getStart() + i; // Token position in the tokens vector
        uint16_t correct_token = tokens->at(token_positon);

        PositionResult* position_output = nullptr;
        if(custom_params.output_writer_type == OutputWriterType::FULL){
            position_output = new PositionFullResult(token_data, correct_token);
        } else if(custom_params.output_writer_type == OutputWriterType::TOP_K){
            position_output = new PositionTopResult(token_data, correct_token, custom_params.top_k);
        }

        T* casted_position_output = dynamic_cast<T*>(position_output);
        result_writer->addPositionResult(*casted_position_output);
    }

    result_writer->writeAndClear();
}



template<typename T>
int extract_probabilities(gpt_params params, CustomParams custom_params){
    assert_output_writer_type<T>(custom_params, custom_params.output_writer_type);

    llama_model * model;
    llama_context * ctx;
    std::tie(model, ctx) = load_model(params);
    
    // Parameters
    const int n_ctx = llama_n_ctx(ctx); // Context size
    const int n_batch = params.n_batch; // Batch size
    const int n_vocab = llama_n_vocab(llama_get_model(ctx)); // Vocab size
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    printf("n_ctx: %d\n", n_ctx);
    printf("n_batch: %d\n", n_batch);
    printf("n_vocab: %d\n", n_vocab);
    printf("add_bos: %d\n", add_bos);

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);
    if (int(tokens.size()) < 2*n_ctx) {
        fprintf(stderr, "%s: you need at least %d tokens to extract probabilities for a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return 1;
    }

    // TODO: generate file name or get it from params
    std::string filename = "output.logits";
    ResultWriter<T> result_writer(filename);
    // check if file exists
    if(std::ifstream(filename).good()){
        printf("File %s already exists. Please remove it and try again.\n", filename.c_str());
        exit(1);
    }
    result_writer.openFile();

    InputIterator<llama_token> input_iterator(&tokens, n_ctx, n_batch);
    ChunkCallback chunk_callback = [&](Chunk chunk){
        printf("Processing chunk %d/%d\n", chunk.getIndex() + 1, input_iterator.getChunksNumber());
        llama_kv_cache_clear(ctx);

        std::vector<float> chunk_logits = get_chunk_logits(chunk, &input_iterator, ctx, n_batch, n_vocab, add_bos);
        write_chunk_logits<T>(&chunk_logits, chunk, &tokens, &result_writer, n_ctx, n_vocab, custom_params);
        // TODO: calculate probabilities and save write them to another file
    };
    input_iterator.iterate(chunk_callback);

    result_writer.closeFile();

    // TODO: save metadata of the experiment

    model_free(ctx, model);

    return 0;
}

int main(int argc, char ** argv) {
    CustomParams custom_params = parse_custom_params(&argc, &argv);
    gpt_params params = init_llama_cpp_params(argc, argv);

    if(custom_params.output_writer_type == OutputWriterType::FULL){
        printf("Extracting full probabilities\n");
        return extract_probabilities<PositionFullResult>(params, custom_params);
    } else if(custom_params.output_writer_type == OutputWriterType::TOP_K){
        printf("Extracting top-k probabilities\n");
        printf("Top-k: %d\n", custom_params.top_k);
        return extract_probabilities<PositionTopResult>(params, custom_params);
    }
}
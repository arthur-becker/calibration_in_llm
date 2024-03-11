#include "common.h"
#include "utils/input_iterator.h"
#include "utils/result_writer.h"
#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"
#include "utils/position_result/position_result.h"
#include "utils/parse_custom_params.h"
#include "utils/llama_cpp_helper.h"
#include "utils/softmax.h"
#include "commands/extract_probabilities.h"
#include "yaml-cpp/yaml.h"
#include "utils/endianness.h"

#include <cstdio>
#include <fstream>
#include <cassert>
#include <filesystem>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

template<typename T>
void assert_output_writer_type(OutputWriterType expected){
    static_assert(std::is_base_of<PositionResult, T>::value, "T must be a subclass of PositionResult");

    // OutputWriterType::FULL -> PositionFullResult
    assert((expected != OutputWriterType::FULL || std::is_base_of_v<PositionFullResult, T>));
    
    // OutputWriterType::TOP_K -> PositionTopResult
    assert((expected != OutputWriterType::TOP_K || std::is_base_of_v<PositionTopResult, T>));

}

template <typename T>
ProbabilitiesExtractor<T>::ProbabilitiesExtractor(int argc, char ** argv, CustomParams custom_params) : LlamaCppHelper(argc, argv){
    this->custom_params = custom_params;
    assert_output_writer_type<T>(custom_params.output_writer_type);
}

template <typename T>
void ProbabilitiesExtractor<T>::tokenize(std::string &prompt){
    this->tokens = ::llama_tokenize(this->getContext(), prompt, this->shouldAddBOS());
    if (int(tokens.size()) < 2*this->getContextSize()) {
        fprintf(stderr, "%s: you need at least %d tokens to extract probabilities for a context of %d\n",
            __func__,
            2*this->getContextSize(),
            this->getContextSize());
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",
            __func__,
            tokens.size());
        exit(1);
    }
    this->input_iterator = new InputIterator<llama_token>(&this->tokens, this->getContextSize(), this->getBatchSize());
}

template <typename T>
void ProbabilitiesExtractor<T>::init_result_writers(std::string output_folder, OutputWriterType output_writer_type){
    // Assume that the executable is run from ./build directory
    std::string output_folder_path = this->getOutputFolderPath(output_folder);
    if (std::filesystem::exists(output_folder_path)) {
        printf("Output folder %s already exists. Please remove it and try again, or change folder name\n", output_folder.c_str());
        exit(1);
    }
    std::filesystem::create_directories(output_folder_path);

    std::string base_file_name = output_folder_path + "/output.";
    if(output_writer_type == OutputWriterType::FULL){
        base_file_name += "full";
    } else if(output_writer_type == OutputWriterType::TOP_K){
        base_file_name += "top";
    }
    else{
        printf("Unknown output writer type\n");
        exit(1);
    }

    this->logits_writer = ResultWriter<T>(base_file_name +  ".logits");
    this->proba_writer = ResultWriter<T>(base_file_name +  ".proba");
}


template <typename T>
void ProbabilitiesExtractor<T>::run(){
    std::string prompt = this->getParams().prompt;
    this->tokenize(prompt);

    this->init_result_writers(this->custom_params.output_folder, this->custom_params.output_writer_type);
    this->logits_writer.openFile();
    this->proba_writer.openFile();

    ChunkCallback chunk_callback = [&](Chunk chunk){
        printf("Processing chunk %d/%d\n", chunk.getIndex() + 1, input_iterator->getChunksNumber());
        llama_kv_cache_clear(this->getContext());

        std::vector<float> chunk_logits = this->get_chunk_logits(chunk);
        this->write_chunk_logits_and_proba(&chunk_logits, chunk);
    };
    this->input_iterator->iterate(chunk_callback);

    this->logits_writer.closeFile();
    this->proba_writer.closeFile();
    this->save_run_info(this->getParams(), this->custom_params);
    printf("Done\n");
}

template <typename T>
std::string ProbabilitiesExtractor<T>::getOutputFolderPath(const std::string& output_folder) const{
    return "../results/" + output_folder;
}

template <typename T>
void ProbabilitiesExtractor<T>::save_run_info(gpt_params params, CustomParams custom_params){
    std::string output_folder = custom_params.output_folder;
    std::string output_folder_path = this->getOutputFolderPath(output_folder);
    std::string run_info_filename = output_folder_path + "/info.yaml";

    YAML::Node run_info;
    run_info["model"] = params.model;
    run_info["dataset_filename"] = params.prompt_file;
    run_info["context_size"] = this->getContextSize();
    run_info["batch_size"] = this->getBatchSize();
    run_info["add_bos"] = this->shouldAddBOS();
    run_info["vocab_size"] = this->getVocabSize();
    run_info["num_tokens"] = this->tokens.size();

    run_info["little_endian"] = isLittleEndian();
    
    // Create a subnode for the custom_params
    YAML::Node output_writer;
    output_writer["top_k"] = custom_params.top_k; // For top-k output writer
    if(custom_params.output_writer_type == OutputWriterType::FULL){
        output_writer["output_writer_type"] = "full";
    } else if(custom_params.output_writer_type == OutputWriterType::TOP_K){
        output_writer["output_writer_type"] = "top-k";
    }
    run_info["output_writer"] = output_writer;

    std::ofstream fout(run_info_filename);
    fout << run_info;
    fout.close();
}

template <typename T>
std::vector<float> ProbabilitiesExtractor<T>::get_chunk_logits(Chunk chunk){
    std::vector<float> chunk_logits(chunk.getSize() * this->getVocabSize());

    BatchCallback batch_callback = [&](Batch batch){
        // save original token and restore it after eval
        const auto original_token = tokens[batch.getStart()];

        // add BOS token for the first batch of each chunk
        if (this->shouldAddBOS() && batch.getIndex() == 0) {
            this->tokens[batch.getStart()] = this->getTokenBOS();
        }

        llama_token* tokens_batch = this->tokens.data() + batch.getStart(); // Pointer to the first token in the batch in the tokens vector
        assert(batch.getSize() == this->getBatchSize());
        llama_pos pos_0 = batch.getIndex() * this->getBatchSize(); // Position in the chunk
        llama_batch batch_data = llama_batch_get_one(tokens_batch, this->getBatchSize(), pos_0, 0);
        if (llama_decode(this->getContext(), batch_data)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            exit(1);
        }

        // restore the original token in case it was set to BOS
        tokens[batch.getStart()] = original_token;

        float * batch_logits = llama_get_logits(this->getContext());
        float * first = (float *) batch_logits; // Pointer to the first logit in the batch
        float * last = first + batch.getSize() * this->getVocabSize(); // Pointer to the first logit of the next batch
        chunk_logits.insert(chunk_logits.end(), first, last);
    };
    input_iterator->iterate(batch_callback, chunk);

    return chunk_logits;
}

template <typename T>
void ProbabilitiesExtractor<T>::write_chunk_logits_and_proba(std::vector<float> * logits, Chunk chunk){
    int second_half_start = this->getContextSize() / 2;
    for(uint32_t i = second_half_start; i < this->getContextSize(); i++){
        float * first = (float *) logits->data() + i * this->getVocabSize(); 
        float * last = first + this->getVocabSize(); 

        std::vector<float> token_data(first, last);

        uint32_t token_positon = chunk.getStart() + i; // Token position in the tokens vector
        uint16_t correct_token = tokens.at(token_positon);

        PositionFullResult logits_full(token_data, correct_token);
        PositionFullResult proba_full = softmax(logits_full);

        // Logits
        if constexpr (std::is_base_of_v<PositionFullResult, T>) {
            this->logits_writer.addPositionResult(std::move(logits_full));
            this->proba_writer.addPositionResult(std::move(proba_full));
        } else if constexpr (std::is_base_of_v<PositionTopResult, T>){
            this->logits_writer.addPositionResult(PositionTopResult(
                std::move(token_data),
                correct_token,
                this->custom_params.top_k));
            this->proba_writer.addPositionResult(PositionTopResult(
                proba_full.getTokenData(),
                correct_token,
                this->custom_params.top_k));
        }
    }

    this->logits_writer.writeAndClear();
    this->proba_writer.writeAndClear();
}

template class ProbabilitiesExtractor<PositionFullResult>;
template class ProbabilitiesExtractor<PositionTopResult>;

int main(int argc, char ** argv) {
    CustomParams custom_params = parse_custom_params(&argc, &argv);

    if(custom_params.output_writer_type == OutputWriterType::FULL){
        printf("Extracting full probabilities\n");

        ProbabilitiesExtractor<PositionFullResult> probabilities_extractor(argc, argv, custom_params);
        probabilities_extractor.run();
    } else if(custom_params.output_writer_type == OutputWriterType::TOP_K){
        printf("Extracting top-k probabilities\n");
        printf("Top-k: %d\n", custom_params.top_k);

        ProbabilitiesExtractor<PositionTopResult> probabilities_extractor(argc, argv, custom_params);
        probabilities_extractor.run();
    }
    return 0;
}
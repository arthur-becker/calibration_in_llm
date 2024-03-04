#include <vector>
#include <common.h>
#include <yaml-cpp/yaml.h>

#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_top_result.h"
#include "utils/result_writer.h"
#include "utils/softmax.h"
#include "utils/parse_custom_params.h"
#include "utils/endianness.h"


/// @brief Mock of the function from the extract_probabilities.cpp
/// @param params 
/// @param custom_params 
void save_run_info(CustomParams custom_params){
    std::string output_folder_path = "../results/" + custom_params.output_folder;
    std::string run_info_filename = output_folder_path + "/info.yaml";

    YAML::Node run_info;
    run_info["model"] = "ggml-model-q4_0.gguf";
    run_info["dataset_filename"] = "test_dataset.txt";
    run_info["context_size"] = 512;
    run_info["batch_size"] = 512;
    run_info["add_bos"] = true;
    run_info["vocab_size"] = 32000;
    run_info["num_tokens"] = 2; // counted from `main` function

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

/// @brief This function generates test data used for testing Python scripts
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char ** argv) {
    std::string output_folder_path_full = "../results/mock_writer_type_full";
    std::string output_folder_path_top = "../results/mock_writer_type_top";

    // Check existence of output folders and create them if they don't exist
    // TODO: check if it compiles for Linux. If not, try using std::filesystem 
    if (std::__fs::filesystem::exists(output_folder_path_full)) {
        printf("Output folder %s already exists. Please remove it and try again, or change folder name\n", output_folder_path_full.c_str());
        exit(1);
    }
    std::__fs::filesystem::create_directories(output_folder_path_full);
    if (std::__fs::filesystem::exists(output_folder_path_top)) {
        printf("Output folder %s already exists. Please remove it and try again, or change folder name\n", output_folder_path_top.c_str());
        exit(1);
    }
    std::__fs::filesystem::create_directories(output_folder_path_top);

    // Mock data
    float token_data1[] = {0.1, 0.2, 0.6, 0.3};
    uint16_t correct_token_1 = 1;

    float token_data2[] = {0.2, 0.5, 0.3, 0.2, 0.8, 0.1};
    uint16_t correct_token_2 = 2;

    // Full results
    PositionFullResult full_logits1(
        std::vector<float>(token_data1, token_data1 + 4),
        correct_token_1
    );
    PositionFullResult full_logits2(
        std::vector<float>(token_data2, token_data2 + 6),
        correct_token_2
    );
    PositionFullResult full_proba1 = softmax(full_logits1);
    PositionFullResult full_proba2 = softmax(full_logits2);

    // Top-k results
    uint16_t k = 3; // top-k
    PositionTopResult top_logits1(
        std::vector<float>(token_data1, token_data1 + 4),
        correct_token_1,
        k
    );
    PositionTopResult top_logits2(
        std::vector<float>(token_data2, token_data2 + 6),
        correct_token_2,
        k
    );

    PositionTopResult top_proba1(
        full_proba1.getTokenData(),
        correct_token_1,
        k
    );
    PositionTopResult top_proba2(
        full_proba2.getTokenData(),
        correct_token_2,
        k
    );

    // Write results to files
    // PositionFullResult
    ResultWriter<PositionFullResult> writer_full_logits(output_folder_path_full + "/output.full.logits");
    writer_full_logits.openFile();
    writer_full_logits.addPositionResult(full_logits1);
    writer_full_logits.addPositionResult(full_logits2);
    writer_full_logits.writeAndClear();
    writer_full_logits.closeFile();

    ResultWriter<PositionFullResult> writer_full_proba(output_folder_path_full + "/output.full.proba");
    writer_full_proba.openFile();
    writer_full_proba.addPositionResult(full_proba1);
    writer_full_proba.addPositionResult(full_proba2);
    writer_full_proba.writeAndClear();
    writer_full_proba.closeFile();

    // PositionTopResult
    ResultWriter<PositionTopResult> writer_top_logits(output_folder_path_top + "/output.top.logits");
    writer_top_logits.openFile();
    writer_top_logits.addPositionResult(top_logits1);
    writer_top_logits.addPositionResult(top_logits2);
    writer_top_logits.writeAndClear();

    ResultWriter<PositionTopResult> writer_top_proba(output_folder_path_top + "/output.top.proba");
    writer_top_proba.openFile();
    writer_top_proba.addPositionResult(top_proba1);
    writer_top_proba.addPositionResult(top_proba2);
    writer_top_proba.writeAndClear();

    // Save run info
    CustomParams custom_params_full;
    custom_params_full.output_folder = "mock_writer_type_full";
    custom_params_full.output_writer_type = OutputWriterType::FULL;
    custom_params_full.top_k = 0;
    save_run_info(custom_params_full);

    CustomParams custom_params_top;
    custom_params_top.output_folder = "mock_writer_type_top";
    custom_params_top.output_writer_type = OutputWriterType::TOP_K;
    custom_params_top.top_k = k;
    save_run_info(custom_params_top);

    return 0;
}

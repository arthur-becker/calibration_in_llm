LLAMA_CPP_URL=$1

# Ask for LLAMA_CPP_URL if not provided until the user provides a correct URL. If the user
# does not provide anything, fill in the default URL: http://localhost:8000
if [ -z "$LLAMA_CPP_URL" ]; then
        echo "\nYou did not provide a URL to the llama-cpp server. Please enter a valid URL."
        read -p "Enter the URL to the llama-cpp server [http://localhost:8000]: " LLAMA_CPP_URL
        if [ -z "$LLAMA_CPP_URL" ]; then
                LLAMA_CPP_URL="http://localhost:8000"
        fi
fi
echo "\nLLAMA_CPP_URL: $LLAMA_CPP_URL"

# Ask user to provide a list of tasks to evaluate the model on, separated by commas. If none are provided, assume the default task.
echo "\nPlease provide a list of tasks to evaluate the model on, separated by commas. If none are provided, the default task will be used."
read -p "Enter the tasks to evaluate the model on [gsm8k_cot_self_consistency_small]: " TASKS
if [ -z "$TASKS" ]; then
        TASKS="gsm8k_cot_self_consistency_small"
fi

# Ask user to provide the output directory for the evaluation results until the user provides a valid directory
echo "\nPlease provide the output directory for the evaluation results."
read -p "Enter the output directory for the evaluation results: " OUTPUT_DIR
while [ ! -d "$OUTPUT_DIR" ]; do
        echo "The directory $OUTPUT_DIR does not exist. Please enter a valid directory."
        read -p "Enter the output directory for the evaluation results: " OUTPUT_DIR
done
echo "\nOUTPUT_DIR: $OUTPUT_DIR\n\n"


# Run the evaluation harness with the given parameters
echo "\n\nRunning the evaluation harness with the following parameters:"
echo "> python3 -m lm_eval --model gguf --tasks $TASKS --model_args base_url=$LLAMA_CPP_URL --output_path $OUTPUT_DIR \n\n"
lm_eval --model gguf \
        --tasks $TASKS \
        --model_args base_url=$LLAMA_CPP_URL
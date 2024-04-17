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

echo "\n\nRunning the evaluation harness with the following parameters:"
echo "> python3 -m lm_eval --model gguf --tasks mmlu_flan_cot_fewshot --model_args base_url=$LLAMA_CPP_URL \n\n"
lm_eval --model gguf \
        --tasks mmlu_flan_cot_fewshot \
        --model_args base_url=$LLAMA_CPP_URL
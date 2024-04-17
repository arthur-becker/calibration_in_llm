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

# Ask if all the layers have to be put on the GPU
read -p "How many layers do you want to put on the GPU? (0 = no, -1=all) [-1]: " N_GPU_LAYERS
if [ -z "$N_GPU_LAYERS" ]; then
        N_GPU_LAYERS=-1
fi

echo "\n\nRunning the evaluation harness with the following parameters:"
echo "> python3 -m lm_eval --model gguf --tasks gsm8k_cot_self_consistency --model_args base_url=$LLAMA_CPP_URL \n\n"
lm_eval --model gguf \
        --tasks gsm8k_cot_self_consistency \
        --model_args base_url=$LLAMA_CPP_URL \
        --n_gpu_layers $N_GPU_LAYERS
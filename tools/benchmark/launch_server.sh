# Read the path to the model
MODEL_PATH=$1
while true; do
    if [ -z "$MODEL_PATH" ]; then
        echo "You did not provide a path to the model as the first argument or as a response to the quesion. Please enter a valid path."
    elif [ -d "$MODEL_PATH" ]; then
        echo "The path is a directory. The files in the directory are:\n"
        ls $MODEL_PATH
    elif [ -f "$MODEL_PATH" ]; then
        break
    else
        echo "The path is not valid. Please enter a valid path."
    fi
    echo "\n"
    read -p "Enter the path to the GGUF model: " MODEL_PATH
done
echo "\nModel path: $MODEL_PATH"
echo "\n\n"

# Read the path to the calibrator
CALIBRATOR_PATH=$2
if [ -z "$CALIBRATOR_PATH" ]; then
    echo "You did not provide a path to the calibrator as the second argument or as a response to the quesion. Please enter a valid path."
    read -p "Enter the path to the calibrator: " CALIBRATOR_PATH
    while true; do
        if [ -z "$CALIBRATOR_PATH" ]; then
            echo "You did not provide a path to the calibrator. Continuing without a calibrator.\n\n"
            break
        elif [ -d "$CALIBRATOR_PATH" ]; then
            echo "The path is a directory. The files in the directory are:\n"
            ls $CALIBRATOR_PATH
        elif [ -f "$CALIBRATOR_PATH" ]; then
            echo "Calibrator path: $CALIBRATOR_PATH \n\n"
            break
        else
            echo "The path is not valid. Please enter a valid path or let it empty."
        fi
        echo "\n"
        read -p "Enter the path to the calibrator: " CALIBRATOR_PATH
    done
fi

# Ask if all the layers have to be put on the GPU
read -p "How many layers do you want to put on the GPU? (0 = no, -1=all) [-1]: " N_GPU_LAYERS
if [ -z "$N_GPU_LAYERS" ]; then
        N_GPU_LAYERS=-1
fi


echo "Starting the server with the following parameters:"

if [ -z "$CALIBRATOR_PATH" ]; then
    echo "> python3 -m llama_cpp.server --model $MODEL_PATH --n_gpu_layers $N_GPU_LAYERS \n\n"
    python3 -m llama_cpp.server \
        --model $MODEL_PATH \
        --n_gpu_layers $N_GPU_LAYERS
else
    echo "> python3 -m llama_cpp.server --model $MODEL_PATH --calibrator_path $CALIBRATOR_PATH --n_gpu_layers $N_GPU_LAYERS \n\n"
    python3 -m llama_cpp.server \
        --model $MODEL_PATH \
        --calibrator_path $CALIBRATOR_PATH \
        --n_gpu_layers $N_GPU_LAYERS
fi
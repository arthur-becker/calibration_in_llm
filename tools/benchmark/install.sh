### This script installs the necessary dependencies for the benchmarking tools.
### 
### Instead of installing fron requirements.txt, we install from git repos.
### The reason is that we probably need to specify some environmental variables
### to enable CUDA support, and we can't do that in requirements.txt.

LLAMA_CPP_PYTHON_REPO="https://github.com/arthur-becker/calibrated-llama-cpp-python.git@add-calibration"
LM_EVALUATION_HARNESS_REPO="https://github.com/arthur-becker/lm-evaluation-harness.git@llama-temperature-sampling"

pip install git+$LLAMA_CPP_PYTHON_REPO
echo "LLAMA_CPP_PYTHON installed\n"

pip install 'llama-cpp-python[server]'

pip install git+$LM_EVALUATION_HARNESS_REPO
echo "LM_EVALUATION_HARNESS installed\n"

echo "\nAll dependencies installed"
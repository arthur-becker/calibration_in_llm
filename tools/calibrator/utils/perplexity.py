from utils.position_result import PositionResult 
from typing import Generator 
import numpy as np
import math

def perplexity(y_true: np.ndarray, y_prob: np.ndarray) -> np.float32:
    """
    This implementation of perplexity is based on the one used in `llama.cpp`:

    In `llama.cpp/examples/perplexity/perplexity.cpp`:
    ```c++
    // perplexity is e^(average negative log-likelihood)
    ```

    proba: list of `PositionResult`'s containing probabilities
    """

    # Indexes of y_true that are 1
    indexes_pos = np.where(y_true == 1)[0]
    y_prob_pos = y_prob[indexes_pos]

    n = len(y_prob_pos)
    nll = np.log(y_prob_pos) * (-1) # negative log likelihood
    
    nll_avg = sum(nll)
    ppl = math.exp(nll_avg/n) # perplexity
       
    return ppl
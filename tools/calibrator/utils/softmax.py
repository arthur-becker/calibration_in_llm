from utils.position_result import PositionFullResult
import numpy as np

def softmax(position_result: PositionFullResult) -> np.array: 
    """
    Calculate the softmax of the logits

    Args:
    - logits: np.array of shape (num_classes,)

    Returns:
    - np.array of shape (num_classes,)
    """
    logits = position_result.token_data

    exps = np.exp(logits - np.max(logits))
    result = exps / np.sum(exps)

    return result

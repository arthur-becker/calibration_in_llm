from utils.position_result import PositionResult
import numpy as np

def softmax(position_result: PositionResult) -> np.array: 
    """
    Calculate the softmax of the logits

    Args:
    - position_result: PositionResult

    Returns:
    - result: numpy array with the softmax of the logits
    """
    logits = position_result.get_token_data()

    exps = np.exp(logits - np.max(logits))
    result = exps / np.sum(exps)

    return result

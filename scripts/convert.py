import numpy as np
from position_result import PositionResult

def position_result_to_numpy(proba: list[PositionResult]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of `PositionResult` to a tuple of numpy arrays (y_true, y_prob) that can be used in sklearn

    Args:
    - proba: list of `PositionResult`'s containing probabilities

    Returns:
    - y_true: numpy array with the true labels
    - y_prob: numpy array with the predicted probabilities
    """

    y_prob = np.array([], dtype=np.float32)
    for result in proba:
        y_prob = np.append(y_prob, result.get_token_data())

    y_true = np.array([], dtype=np.float32)
    for result in proba:
        y_true = np.append(y_true, [i == result.get_correct_token() for i in range(len(result.get_token_data()))])
    
    assert y_true.shape == y_prob.shape

    return y_true, y_prob

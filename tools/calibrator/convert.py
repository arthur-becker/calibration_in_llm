import numpy as np
from position_result import PositionResult

def position_result_to_numpy(results: list[PositionResult]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of `PositionResult` to a tuple of numpy arrays (y_true, y_prob) that can be used in sklearn

    Args:
    - proba: list of `PositionResult`'s containing logits or probabilities

    Returns:
    - y_true: numpy array with the true labels
    - y_value: numpy array with the predicted probabilities
    """

    size = results[0].get_token_data().shape[0]

    y_value = np.array([], dtype=np.float32)
    y_true = np.array([], dtype=np.float32)
    for result in results:
        y_value = np.append(y_value, result.get_token_data())
        y_true = np.append(y_true, [i == result.get_correct_token() for i in range(len(result.get_token_data()))])

        assert result.get_token_data().shape[0] == size
        # This implementations assumes that every PositionResult has the same number of tokens.
        # If this is not the case, the sizes should be returned as a list
        #
        # Except for that, the other places may need to be changed to handle the case where the sizes are different
    
    assert y_true.shape == y_value.shape

    return y_true, y_value, size

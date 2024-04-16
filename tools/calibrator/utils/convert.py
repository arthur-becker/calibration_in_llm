import numpy as np
from utils.position_result import PositionResult, PositionFullResult, PositionTopResult
from utils.softmax import softmax

def position_result_to_numpy(results: list[PositionResult], show_logs=False) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Convert a list of `PositionResult` to a tuple of numpy arrays (y_true, y_prob) that can be used in sklearn

    Args:
    - proba: list of `PositionResult`'s containing logits or probabilities

    Returns:
    - y_true: numpy array with the true labels
    - y_value: numpy array with the predicted probabilities
    """

    size = results[0].get_token_data().shape[0]
    results_len = len(results)

    y_value = np.empty((size * results_len), dtype=np.float32)
    y_true = np.empty((size * results_len), dtype=np.float32)
    for i in range(results_len):
        result = results[i]
        token_data = result.get_token_data()
        correct_token = result.get_correct_token()

        y_value[i*size:(i+1)*size] = token_data
        y_true[i*size:(i+1)*size] = [i == correct_token for i in range(size)]

        assert result.get_token_data().shape[0] == size
        # This implementations assumes that every PositionResult has the same number of tokens.
        # If this is not the case, the sizes should be returned as a list
        #
        # Except for that, the other places may need to be changed to handle the case where the sizes are different

        if show_logs:
            if i % 100 == 0:
                print(f"position_result_to_numpy: {i}/{results_len}")
    
    assert y_true.shape == y_value.shape

    return y_true, y_value, size

def logits_to_proba(logits_position_result: PositionResult) -> PositionResult:
    if isinstance(logits_position_result, PositionTopResult):
        proba = softmax(logits_position_result)
        proba_position_result = PositionTopResult(proba, logits_position_result.n)
        return proba_position_result
    elif isinstance(logits_position_result, PositionFullResult):
        proba = softmax(logits_position_result)
        proba_position_result = PositionFullResult(proba, logits_position_result.get_correct_token())
        return proba_position_result
    else:
        raise ValueError(f'Unknown position result type: {type(logits_position_result)}')

import numpy as np

def normalize(y_prob: np.array, position_size: int):
    """
    Normalize the probabilities of the model.

    Args:
    - y_prob: numpy array with the predicted probabilities
    - position_size: int, the number of tokens in a position

    Returns:
    - y_norm: numpy array with the normalized probabilities
    """
    y_norm = y_prob.copy()
    for i in range(0, len(y_prob), position_size):
        y_norm[i:i+position_size] = y_prob[i:i+position_size] / np.sum(y_prob[i:i+position_size])
    return y_norm

def denormalize(y_prob: np.array, y_ref: np.array, position_size: int):
    """
    Denormalize the probabilities of the model.

    Args:
    - y_prob: numpy array with the predicted probabilities
    - y_ref: numpy array with the reference probabilities
    - position_size: int, the number of tokens in a position

    Returns:
    - y_denorm: numpy array with the denormalized probabilities
    """
    y_denorm = y_prob.copy()
    for i in range(0, len(y_prob), position_size):
        y_denorm[i:i+position_size] = y_prob[i:i+position_size] / np.sum(y_prob[i:i+position_size]) * np.sum(y_ref[i:i+position_size])
    return y_denorm


if __name__ == '__main__':
    print('\n\nSTARTING NORMALIZATION TEST...')

    y = np.array([0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    print(f'y: {y}')

    y_norm = normalize(y, 3)
    print(f'y_norm: {y_norm}')

    y_denorm = denormalize(y_norm, y, 3)
    print(f'y_denorm: {y_denorm}')
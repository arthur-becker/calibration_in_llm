import numpy as np

bin

def ece(y_prob, y_true, n_bins=15):
    """
    Expected Calibration Error
    """

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_prob, axis=1)
    accuracies = y_prob[np.arange(y_prob.shape[0]), y_true]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece

if __name__ == "__main__":
    # Example: https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d

    # Example usage
    y_prob = np.array([0.1, 0.9, 0.8, 0.4, 0.5, 0.6, 0.7, 0.3, 0.2, 0.99])
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])


    

import argparse
from experiment_info import RunInfo
from position_result import PositionResult
from matplotlib import pyplot as plt
from convert import position_result_to_numpy
from sklearn.calibration import calibration_curve

# TODO: visualize the distribution of the output data


def visualize_hist(y_true, y_prob, show=False, save=True, save_name=None, bins=10):
    plt.hist(y_prob, bins=bins)

    if show:
        plt.show()

    if save:
        if save_name is None:
            raise ValueError('save_name should be provided if save is True')
        plt.savefig(save_name)

def visualize_calibration_curve(y_true, y_prob, show=False, save=True, save_name=None, bins=10):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Calibration curve')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve')
    plt.legend()

    if show:
        plt.show()

    if save:
        if save_name is None:
            raise ValueError('save_name should be provided if save is True')
        plt.savefig(save_name)


if __name__ == '__main__':
    # Preparation
    parser = argparse.ArgumentParser(description='Visualize the output data')
    parser.add_argument('llama_cpp_run', 
        type=str, 
        help='The name of the folder in `results/` where llama.cpp stored the output')
    args = parser.parse_args()

    run_info = RunInfo(args.llama_cpp_run)

    proba_position_result_list : PositionResult = list(run_info.proba_reader.read())
    y_true, y_prob = position_result_to_numpy(proba_position_result_list)

    # Visualization
    print('\n\nVisualizing the output data...')
    visualize_hist(y_true, y_prob, show=True, save=False)
    visualize_calibration_curve(y_true, y_prob, show=True, save=False)
    print('Visualization finished.')
    
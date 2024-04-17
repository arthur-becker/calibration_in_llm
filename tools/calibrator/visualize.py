
import argparse
from experiment_info import RunInfo
from utils.position_result import PositionResult
from matplotlib import pyplot as plt
from utils.convert import position_result_to_numpy, logits_to_proba
from sklearn.calibration import calibration_curve

FIG_SIZE = (7, 7)

def visualize_hist(y_true, y_prob, show=False, save=True, save_name=None, bins=10):
    plt.figure(figsize=FIG_SIZE)
    plt.hist(y_prob, bins=bins, edgecolor='black')
    plt.yscale('symlog')
    

    if show:
        plt.show()

    if save:
        if save_name is None:
            raise ValueError('save_name should be provided if save is True')
        plt.savefig(save_name)
        plt.close()

def visualize_calibration_curve(y_true, y_prob, show=False, save=True, save_name=None, bins=10):
    plt.figure(figsize=FIG_SIZE)
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
        plt.close()

def visualize_changes(X, y_lists, labels, xlabel="X", ylabel="Y", show=False, save=True, save_name=None):
    plt.figure(figsize=FIG_SIZE)

    assert len(y_lists) == len(labels), 'y_lists and labels should have the same length'

    for i in range(len(y_lists)):
        y = y_lists[i]
        assert len(X) == len(y), 'X and y should have the same length'
        plt.plot(X, y, label=labels[i])

    y_all = []
    X_all = []
    for i in range(len(y_lists)):
        y_all += y_lists[i]
        X_all += X
    

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(X_all, y_all, color='red', s=50, zorder=5)
    plt.legend(loc='upper right') 
    plt.tight_layout()

    if show:
        plt.show()

    if save:
        if save_name is None:
            raise ValueError('save_name should be provided if save is True')
        plt.savefig(save_name)
        plt.close()


if __name__ == '__main__':
    # Preparation
    parser = argparse.ArgumentParser(description='Visualize the output data')
    parser.add_argument('llama_cpp_run', 
        type=str, 
        help='The name of the folder in `outputs/extractor/` where llama.cpp stored the output')
    args = parser.parse_args()

    run_info = RunInfo(args.llama_cpp_run)

    i = 0
    proba_position_result_list = []
    for logits in run_info.logits_reader.read():
        proba = logits_to_proba(logits)
        proba_position_result_list.append(proba)
        if i % 100 == 0:
            print(f'Unpacked a position result: {i+1}')
        i += 1
    print(f'proba_position_result_list has been created')

    y_true, y_prob, size = position_result_to_numpy(proba_position_result_list, show_logs=True)
    print(f'Numpy arrays have been created')

    # Visualization
    print('\n\nVisualizing the output data...')
    visualize_hist(y_true, y_prob, show=True, save=False)
    visualize_calibration_curve(y_true, y_prob, show=True, save=False)
    print('Visualization finished.')
    
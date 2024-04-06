import argparse
from position_result import PositionResult
from result_reader import ResultReader
from convert import position_result_to_numpy
from perplexity import perplexity
from experiment_info import RunInfo
import os
from visualize import visualize_hist, visualize_calibration_curve
from sklearn.metrics import brier_score_loss
import numpy as np

def read_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('output_folder',
        type=str,
        help='Path where the output of this script will be saved.')
    parser.add_argument('llama_cpp_run', 
        type=str, 
        help='The name of the folder in `results/` where C++ program stored the output of the run of llama.cpp')
    return parser.parse_args()

def evaluate(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_folder_path: str,):
    """
    Evaluate the model with different calibration types:
    - uncalibrated probabilities
    - probabilities calibrated with Platt scaling
    - probabilities calibrated with isotonic regression
    """

    # Calculate metrics
    ppl = perplexity(y_true, y_prob)
    brier_score = brier_score_loss(y_true, y_prob)

    # Visualize
    visualize_hist(y_true, y_prob, save_name=output_folder_path + 'probabilities_distribution.png')
    visualize_calibration_curve(y_true, y_prob, save_name=output_folder_path + 'calibration_curve.png')

    # Print results
    print(f'Perplexity of uncalibrated probabilities: {ppl}')
    print(f'Brier score of uncalibrated probabilities: {brier_score}')

if __name__ == '__main__':
    # Preparation
    args = read_args()
    output_folder_path = "./../results/" + args.output_folder + "/"
    llama_cpp_run = RunInfo(args.llama_cpp_run)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    else:
        raise ValueError(f'Folder {output_folder_path} already exists. Please change the output folder name.')

    # Evaluation
    print('\n\nSTARTING EVALUATION...')

    # 1. Load the calibration set results and evaluate them
    proba_position_result_list : PositionResult = list(llama_cpp_run.proba_reader.read())
    print(f'Loaded {len(proba_position_result_list)} position results from the calibration set')

    y_cal_true, y_cal_prob = position_result_to_numpy(proba_position_result_list)
    evaluate(y_cal_true, y_cal_prob, output_folder_path)

    print('\nEVALUATION FINISHED. Results are saved in the folder: ', output_folder_path)
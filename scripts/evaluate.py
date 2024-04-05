import argparse
from position_result import PositionResult
from result_reader import ResultReader
from convert import position_result_to_numpy
from perplexity import perplexity
from experiment_info import get_experiment_info
import os
from visualize import visualize_hist, visualize_calibration_curve
from sklearn.metrics import brier_score_loss

def read_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('output_folder', 
        type=str, 
        help='The name of the folder in `results/` where C++ program stored the output')
    return parser.parse_args()

def calculate_metrics(y_true, y_prob):
    ppl = perplexity(y_true, y_prob)
    brier_score = brier_score_loss(y_true, y_prob)
    return ppl, brier_score

def evaluate(
        logits_reader: ResultReader, 
        proba_reader: ResultReader, 
        output_writer_type: str,
        evaluation_output_folder_path: str):
    """
    Evaluate the model with different calibration types:
    - uncalibrated probabilities
    - probabilities calibrated with Platt scaling
    - probabilities calibrated with isotonic regression
    """
    
    assert output_writer_type in ['full']
    # TODO: Add support for 'top'

    proba_position_result_list : PositionResult = list(proba_reader.read())
    print(f'File with probabilities has been read. '
          f'Lenght of proba_position_result_list: {len(proba_position_result_list)}')

    y_true, y_prob = position_result_to_numpy(proba_position_result_list)
    print(f'Converted to numpy arrays. '
          f'y_true.shape: {y_true.shape}, y_prob.shape: {y_prob.shape}')

    # 1. Evaluate uncalibrated probabilities
    print('\n\nEvaluating uncalibrated probabilities...')
    ppl, brier_score = calculate_metrics(y_true, y_prob)
    print(f'Perplexity of uncalibrated probabilities: {ppl}')
    print(f'Brier score of uncalibrated probabilities: {brier_score}')

    visualize_hist(y_true, y_prob, save_name=evaluation_output_folder_path + 'uncalibrated_probabilities_distribution.png')
    visualize_calibration_curve(y_true, y_prob, save_name=evaluation_output_folder_path + 'uncalibrated_calibration_curve.png')

    # 2. Evaluate probabilities calibrated with Platt scaling

    # 3. Evaluate probabilities calibrated with isotonic regression

if __name__ == '__main__':
    # Preparation
    args = read_args()
    output_folder = args.output_folder
    logits_reader, proba_reader, output_writer_type, path = get_experiment_info(output_folder)

    evaluation_output_folder_path = path + 'evaluation/'
    if not os.path.exists(evaluation_output_folder_path):
        os.makedirs(evaluation_output_folder_path)
    else:
        raise ValueError(f'Folder {evaluation_output_folder_path} already exists. Please remove it before running the script.')


    # Evaluation
    print('\n\nSTARTING EVALUATION...')
    evaluate(logits_reader, proba_reader, output_writer_type, evaluation_output_folder_path)
    print('\nEVALUATION FINISHED. Results are saved in the folder: ', evaluation_output_folder_path)
### This script is the main pipeline script that will be used 
### to calibrate the model and evaluate it.
import argparse
import os
from experiment_info import RunInfo
from convert import position_result_to_numpy
from evaluate import evaluate

def read_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('output_folder',
        type=str,
        help='Path where the output of this script will be saved.')
    parser.add_argument('calibration_set_run', 
        type=str, 
        help='The name of the folder in `results/` where C++ program stored the output of the run on the calibration set')
    parser.add_argument('test_set_run',
        type=str,
        help='The name of the folder in `results/` where C++ program stored the output of the run on the test set')
    return parser.parse_args()

if __name__ == "__main__":
    print('\n\nSTARTING MAIN PIPELINE...')

    args = read_args()

    if not os.path.exists(f'./../results/{args.output_folder}'):
        os.makedirs(f'./../results/{args.output_folder}')
    else:
        raise ValueError(f'Folder ./../results/{args.output_folder} already exists. Please change the output folder name.')
    
    # 1. Evaluate the calibration set before calibration
    print('1. Evaluating the calibration set before calibration...')
    print('1.1. Loading the calibration set...')
    calibration_set_run = RunInfo(args.calibration_set_run)
    position_result_proba = list(calibration_set_run.proba_reader.read())
    y_true_cal, y_prob_cal = position_result_to_numpy(position_result_proba)
    save_path = f'./../results/{args.output_folder}/uncalibrated_calibration_set_'
    ppl, brier_score = evaluate(y_true_cal, y_prob_cal, save_path)
    print(f'Calibraion set (uncalibrated): Perplexity={ppl}, Brier score={brier_score}')

    # 2. Evaluate the test set
    print('2. Evaluating the test set before calibration...')
    print('2.1. Loading the test set...')
    test_set_run = RunInfo(args.test_set_run)
    position_result_proba = list(test_set_run.proba_reader.read())
    y_true_test, y_prob_test = position_result_to_numpy(position_result_proba)
    save_path = f'./../results/{args.output_folder}/uncalibrated_test_set_'
    ppl, brier_score = evaluate(y_true_test, y_prob_test, save_path)
    print(f'Test set (uncalibrated): Perplexity={ppl}, Brier score={brier_score}')

    # TODO: Calibrate with isotonic regression

    # TODO: Calibrate with Platt Scaling

    print('MAIN PIPELINE FINISHED.')
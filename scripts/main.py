### This script is the main pipeline script that will be used 
### to calibrate the model and evaluate it.
import argparse
import os
from experiment_info import RunInfo
from convert import position_result_to_numpy
from evaluate import evaluate
from sklearn.isotonic import IsotonicRegression
import numpy as np
from normalize import normalize, denormalize

def read_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('calibration_set_run', 
        type=str, 
        help='The name of the folder in `results/` where C++ program stored the output of the run on the calibration set')
    parser.add_argument('test_set_run',
        type=str,
        help='The name of the folder in `results/` where C++ program stored the output of the run on the test set')
    parser.add_argument('output_folder',
        type=str,
        help='Path where the output of this script will be saved.')
    parser.add_argument('-normalize_input', 
        action='store_true',
        help='Normalize the input probabilities before calibration')
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
    y_true_cal, y_prob_cal, position_size_cal = position_result_to_numpy(position_result_proba)

    if args.normalize_input:
        print('1.1.1. Normalizing the input probabilities...')
        y_prob_cal = normalize(y_prob_cal, position_size_cal)
    else:
        print('1.1.1. Skipping normalization of the input probabilities...')

    print('1.2. Evaluating...')

    save_path = f'./../results/{args.output_folder}/uncalibrated_calibration_set_'
    ppl, brier_score = evaluate(y_true_cal, y_prob_cal, save_path)
    print(f'Calibraion set (uncalibrated): Perplexity={ppl}, Brier score={brier_score}')

    avg_sum_cal = np.mean([np.sum(y_prob_cal[i:i+position_size_cal]) for i in range(0, len(y_prob_cal), position_size_cal)])
    print(f'-- average sum: {avg_sum_cal}')
    min_sum_cal = np.min([np.sum(y_prob_cal[i:i+position_size_cal]) for i in range(0, len(y_prob_cal), position_size_cal)])
    print(f'-- min sum: {min_sum_cal}')
    max_sum_cal = np.max([np.sum(y_prob_cal[i:i+position_size_cal]) for i in range(0, len(y_prob_cal), position_size_cal)])
    print(f'-- max sum: {max_sum_cal}')

    # 2. Evaluate the test set
    print('2. Evaluating the test set before calibration...')
    print('2.1. Loading the test set...')
    test_set_run = RunInfo(args.test_set_run)
    position_result_proba = list(test_set_run.proba_reader.read())
    y_true_test, y_prob_test, position_size_test = position_result_to_numpy(position_result_proba)
    if args.normalize_input:
        print('2.1.1. Normalizing the input probabilities...')
        y_prob_test = normalize(y_prob_test, position_size_test)
    else:
        print('2.1.1. Skipping normalization of the input probabilities...')

    print('2.2. Evaluating...')
    save_path = f'./../results/{args.output_folder}/uncalibrated_test_set_'
    ppl, brier_score = evaluate(y_true_test, y_prob_test, save_path)
    print(f'Test set (uncalibrated): Perplexity={ppl}, Brier score={brier_score}')

    avg_sum_test = np.mean([np.sum(y_prob_test[i:i+position_size_test]) for i in range(0, len(y_prob_test), position_size_test)])
    print(f'-- average sum: {avg_sum_test}')
    min_sum_test = np.min([np.sum(y_prob_test[i:i+position_size_test]) for i in range(0, len(y_prob_test), position_size_test)])
    print(f'-- min sum: {min_sum_test}')
    max_sum_test = np.max([np.sum(y_prob_test[i:i+position_size_test]) for i in range(0, len(y_prob_test), position_size_test)])
    print(f'-- max sum: {max_sum_test}')

    # 3. Calibrate with isotonic regression
    print('3. Calibrating with isotonic regression...')
    print('3.1. Fitting the isotonic regression model...')
    X_probs_cal = y_prob_cal.reshape(-1, 1)
    min_prob = np.min(X_probs_cal) # Set minimum value to avoid division by 0 when calculating perplexity
    iso_regressor = IsotonicRegression(out_of_bounds='clip', y_min=min_prob, y_max=1)
    iso_regressor.fit(X_probs_cal, y_true_cal) # X_probs_cal.ravel() ?

    print('3.2. Calibrating the calibration set...')
    y_prob_cal_transformed = iso_regressor.transform(X_probs_cal)
    y_prob_cal_transformed = denormalize(y_prob_cal_transformed, y_prob_cal, position_size_cal)

    print('3.3. Evaluating the calibration set after calibration...')
    save_path = f'./../results/{args.output_folder}/calibrated_calibration_set_'
    ppl, brier_score = evaluate(y_true_cal, y_prob_cal_transformed, save_path)
    print(f'Calibration set (isotonic regression): Perplexity={ppl}, Brier score={brier_score}')

    print('3.4. Calibrating the test set...')
    X_prob_test = y_prob_test.reshape(-1, 1)
    y_prob_test_transformed = iso_regressor.transform(X_prob_test)
    y_prob_test_transformed = denormalize(y_prob_test_transformed, y_prob_test, position_size_test) 

    print('3.5. Evaluating the test set after calibration...')
    save_path = f'./../results/{args.output_folder}/calibrated_test_set_'
    ppl, brier_score = evaluate(y_true_test, y_prob_test_transformed, save_path)
    print(f'Test set (isotonic regression): Perplexity={ppl}, Brier score={brier_score}')
    
    print('4. Calibrating with Platt Scaling...')
    # TODO: Calibrate with Platt Scaling

    print('MAIN PIPELINE FINISHED.')
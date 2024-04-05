import argparse
from position_result import PositionResult, PositionFullResult, PositionTopResult
from result_reader import ResultReader
from convert import position_result_to_numpy
from perplexity import perplexity
from experiment_info import get_experiment_info

def read_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('output_folder', 
        type=str, 
        help='The name of the folder in `results/` where C++ program stored the output')
    return parser.parse_args()

def calculate_metrics(y_true, y_prob):
    ppl = perplexity(y_true, y_prob)
    return ppl

def evaluate(
        logits_reader: ResultReader, 
        proba_reader: ResultReader, 
        output_writer_type: str):
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
    ppl = calculate_metrics(y_true, y_prob)
    print(f'Perplexity of uncalibrated probabilities: {ppl}')

    # 2. Evaluate probabilities calibrated with Platt scaling

    # 3. Evaluate probabilities calibrated with isotonic regression

if __name__ == '__main__':
    # Preparation
    args = read_args()
    output_folder = args.output_folder
    logits_reader, proba_reader, output_writer_type = get_experiment_info(output_folder)

    # Evaluation
    print('\n\nEvaluating the model...')
    evaluate(logits_reader, proba_reader, output_writer_type)
    print('Evaluation finished.')
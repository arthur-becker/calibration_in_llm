import yaml
import argparse
import os
import struct
from typing import Generator
from position_result import PositionResult, PositionFullResult, PositionTopResult
import cpp_constants
from result_reader import ResultReader

def read_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def read_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('output_folder', 
        type=str, 
        help='The name of the folder in `results/` where C++ program stored the output')
    return parser.parse_args()

if __name__ == '__main__':
    # Preparation
    args = read_args()

    path = f'./../results/{args.output_folder}/'
    if not os.path.exists(path):
        raise ValueError(f'Path {path} does not exist')
    run_info = read_yaml(path + 'info.yaml')
    
    output_writer_type = None
    if run_info['output_writer']['output_writer_type'] == 'top-k':
        output_writer_type = 'top'
    elif run_info['output_writer']['output_writer_type'] == 'full':
        output_writer_type = 'full'
    else:
        raise ValueError(f'Unknown output_writer_type: {run_info["output_writer"]["output_writer_type"]}')

    logits_filename = f'output.{output_writer_type}.logits'
    proba_filename = f'output.{output_writer_type}.proba'

    little_endian = run_info['little_endian']
    logits_reader = ResultReader(path + logits_filename, little_endian)
    proba_reader = ResultReader(path + proba_filename, little_endian)

    # Evaluation
    print('Evaluating the model...')
import os
import yaml
from result_reader import ResultReader
import argparse

def read_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def get_experiment_info(output_folder_name: str) -> dict:
    """
    Read the experiment information from the info.yaml file.
    """

    path = f'./../results/{output_folder_name}/'
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

    return logits_reader, proba_reader, output_writer_type, path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('output_folder', 
        type=str, 
        help='The name of the folder in `results/` where C++ program stored the output')
    args = parser.parse_args()

    logits_reader, proba_reader, output_writer_type, path = get_experiment_info(args.output_folder)
    print("\nResults from get_experiment_info():")
    print(f'logits_reader: {logits_reader}')
    print(f'proba_reader: {proba_reader}')
    print(f'output_writer_type: {output_writer_type}')
    print(f'path: {path}')
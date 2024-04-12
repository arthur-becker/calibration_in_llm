import os
import yaml
from result_reader import ResultReader
import argparse

def read_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

class RunInfo:
    def __init__(self, output_folder_name: str):
        self.path = f'./../../outputs/extractor/{output_folder_name}/'
        if not os.path.exists(self.path):
            raise ValueError(f'Path {self.path} does not exist')
        
        run_info = read_yaml(self.path + 'info.yaml')
        
        self.output_writer_type = None
        if run_info['output_writer']['output_writer_type'] == 'top-k':
            self.output_writer_type = 'top'
        elif run_info['output_writer']['output_writer_type'] == 'full':
            self.output_writer_type = 'full'
        else:
            raise ValueError(f'Unknown output_writer_type: {run_info["output_writer"]["output_writer_type"]}')

        logits_filename = f'output.{self.output_writer_type}.logits'
        proba_filename = f'output.{self.output_writer_type}.proba'
        self.little_endian = run_info['little_endian']

        self.logits_reader = ResultReader(self.path + logits_filename, self.little_endian)
        self.proba_reader = ResultReader(self.path + proba_filename, self.little_endian)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('output_folder', 
        type=str, 
        help='The name of the folder in `outputs/extractor/` where C++ program stored the output')
    args = parser.parse_args()

    run_info = RunInfo(args.output_folder)
    print("\nResults from get_experiment_info():")
    print(f'logits_reader: {run_info.logits_reader}')
    print(f'proba_reader: {run_info.proba_reader}')
    print(f'output_writer_type: {run_info.output_writer_type}')
    print(f'path: {run_info.path}')
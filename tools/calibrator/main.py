### This script is the main pipeline script that will be used 
### to calibrate the model and evaluate it.
import argparse
import os
from experiment_info import RunInfo
from utils.convert import position_result_to_numpy, logits_to_proba
from evaluate import evaluate
from sklearn.isotonic import IsotonicRegression
import numpy as np
import joblib
import yaml
from dataclasses import dataclass, field
from typing import List

def read_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('calibration_set_run', 
        type=str, 
        help='The name of the folder in `outputs/extractor/` where C++ program stored the output of the run on the calibration set')
    parser.add_argument('test_set_run',
        type=str,
        help='The name of the folder in `outputs/extractor/` where C++ program stored the output of the run on the test set')
    parser.add_argument('output_folder',
        type=str,
        help='Path where the output of this script will be saved.')
    parser.add_argument('--calibration_steps',
        type=int,
        default=1,
        help='The script will split the logits in `calibration_steps` parts and create `calibration_steps`'
            ' different calibration models each being calibrated on different amount of input tokens. This is important'
            ' to understand the necessary amount of data to calibrate the model properly.')
    return parser.parse_args()

@dataclass
class SequenceData:
    y_true: np.ndarray
    y_value: np.ndarray
    position_size: np.ndarray
    X_value: np.ndarray = None

@dataclass
class CalibrationStepStats:
    """
    This class is used to store the statistics of a calibration step that
    may be required to compare the performance of the model over calibration steps.
    """

    num_logits: List[int] = field(default_factory=list) # Number of logits used for calibration
    ppl: List[float] = field(default_factory=list)
    brier_score: List[float] = field(default_factory=list)


    def add(self, num_logits: int, ppl: float, brier_score: float):
        self.num_logits.append(num_logits)
        self.ppl.append(ppl)
        self.brier_score.append(brier_score)

    def __dict__(self):
        result = []

        for i in range(len(self.num_logits)):
            result.append({
                'calibration_step': i,
                'num_logits': self.num_logits[i],
                'perplexity': self.ppl[i],
                'brier_score': self.brier_score[i],
                'perplexity_improvement': self.ppl[i] < self.ppl[0],
                'brier_score_improvement': self.brier_score[i] < self.brier_score[0]
            })
        return result

class MainPipeline:
    def __init__(
            self,
            calibration_set_run: RunInfo,
            test_set_run: RunInfo,
            output_folder: str,
            calibration_steps: int = 1
        ) -> None:
        assert calibration_steps > 0, 'The number of calibration steps must be greater than 0.'

        self.calibration_set_run = calibration_set_run
        self.test_set_run = test_set_run
        self.calibration_steps = calibration_steps

        self.cal_proba : SequenceData = None
        self.test_proba : SequenceData = None

        # Isotonic regression statistics
        self.iso_stats_cal = CalibrationStepStats() 
        self.iso_stats_test = CalibrationStepStats()

        # Logistic regression statistics 
        self.log_stats_cal = CalibrationStepStats() 
        self.log_stats_test = CalibrationStepStats() 

        self.output_folder_path = f'./../../outputs/calibrator/{output_folder}'
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
        else:
            raise ValueError(f'Folder {self.output_folder_path} already exists. Please change the output folder name.')
        

    def run(self):
        print('STARTING MAIN PIPELINE...')
        self.cal_proba : SequenceData = self._load_data(self.calibration_set_run)
        self.test_proba : SequenceData = self._load_data(self.test_set_run)

        for i in range(self.calibration_steps + 1):
            """
            We add one more step because the step 0 evaluates the uncalibrated model.
            """
            print(f'\n\n\nCalibration step: {i}/{self.calibration_steps}')
            self._calibration_step(i)

        print('\n\n')

        print('Saving the results...')
        results = {
            "calibration_set": {
                "path": self.calibration_set_run.path,
                "isotonic": self.iso_stats_cal.__dict__(),
                "logistic": None
            },
            "test_set": {
                "path": self.test_set_run.path,
                "isotonic": self.iso_stats_test.__dict__(),
                "logistic": None
            }
        }
        with open(f'{self.output_folder_path}/results.yaml', 'w') as f:
            yaml.dump(results, f)
        print(f'Results saved in {self.output_folder_path}')

        # TODO: copy the info.yaml file from the calibration set to the output folder

        print('\n\nMAIN PIPELINE FINISHED.')

    def _calibration_step(self, step: int):
        """
        Convention: all the logs written from a calibration step
        should have -- at the beginning of the line.
        """
        num_logits = int(len(self.cal_proba.X_value) * (step) / args.calibration_steps)
        print(f'-- Calibration set size: {num_logits} logits/probabilities.')

        step_folder_path = f'{self.output_folder_path}/step_{step}-{args.calibration_steps}'
        iso_regressor_folder_path = f'{step_folder_path}/isotonic_regressor'
        os.makedirs(iso_regressor_folder_path, exist_ok=True)

        # Effective calibration data for the calibration step i
        X_proba_cal = self.cal_proba.X_value[:num_logits]
        y_true_cal = self.cal_proba.y_true[:num_logits]

        # Isotonic regression
        iso_regressor = None
        if step > 0:
            # Calibrate with isotonic regression
            min_prob = np.min(X_proba_cal) # Set minimum value to avoid division by 0 when calculating perplexity
            iso_regressor = IsotonicRegression(out_of_bounds='clip', y_min=min_prob, y_max=1)
            iso_regressor.fit(X_proba_cal, y_true_cal)
            print(f'-- Isotonic Regressor has been trained. Starting calibration...')

            joblib.dump(iso_regressor, f'{iso_regressor_folder_path}/model.joblib')
            print(f'-- Model saved.')
        self._evaluate(
            step,
            num_logits,
            self.iso_stats_cal,
            self.iso_stats_test,
            iso_regressor_folder_path,
            regressor=iso_regressor)
        
        # TODO: Logistic regression
        print(f'-- Evaluation completed')
        print(f'-- Calibration step {step} completed.')

    def _evaluate(
            self, 
            step: int, 
            num_logits: int,
            stats_cal: CalibrationStepStats,
            stats_test: CalibrationStepStats,
            save_folder_path: str,
            regressor = None
        ):
        subfolder = None
        if isinstance(regressor, IsotonicRegression):
            subfolder = 'isotonic_regressor'
        elif regressor is None and step == 0:
            subfolder = 'uncalibrated'
            print(f'-- Evaluating uncalibrated model...')
        else:
            raise ValueError(f'Unknown regressor type: {type(regressor)}')
        
        # Calibration set
        save_path_cal = f'{save_folder_path}/calibration_set_'
        y_proba_cal = self.cal_proba.y_value
        if step > 0:
            y_proba_cal = regressor.transform(self.cal_proba.X_value)
        ppl, brier_score = evaluate(self.cal_proba.y_true, y_proba_cal, save_path_cal)
        stats_cal.add(num_logits, ppl, brier_score)

        # Test set
        save_path_test = f'{save_folder_path}/test_set_'
        y_proba_test = self.test_proba.y_value
        if step > 0:
            y_proba_test = regressor.transform(self.test_proba.X_value)
        ppl, brier_score = evaluate(self.test_proba.y_true, y_proba_test, save_path_test)
        stats_test.add(num_logits, ppl, brier_score)

    def _load_data(self, run_info: RunInfo) -> SequenceData:
        position_result_proba = [
                logits_to_proba(logits) 
                for logits in run_info.logits_reader.read()
            ]
        y_true, y_proba, position_size = position_result_to_numpy(position_result_proba)
        data = SequenceData(y_true, y_proba, position_size)
        data.X_value = y_proba.reshape(-1, 1)
        return data


if __name__ == "__main__":
    print('Reading arguments...')

    args = read_args()

    pipeline = MainPipeline(
        RunInfo(args.calibration_set_run),
        RunInfo(args.test_set_run),
        args.output_folder,
        args.calibration_steps
    )
    pipeline.run()
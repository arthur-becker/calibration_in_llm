### This script is the main pipeline script that will be used 
### to calibrate the model and evaluate it.
import argparse
import os
from experiment_info import RunInfo
from utils.convert import position_result_to_numpy, logits_to_proba
from utils.inverse_sigmoid import inverse_sigmoid
from utils.normalize import normalize
from visualize import visualize_changes
from evaluate import evaluate
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import yaml
from dataclasses import dataclass, field
from typing import List

import shutil

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
    parser.add_argument('--prob-bins', 
        type=int, 
        default=20,
        help='Number of bins to visualize on the probability histogram or calibration curve. Default is 50.')
    parser.add_argument('--disable_normalization',
        action='store_true',
        help='Disables normalization of the probabilities. Default is disables.',
        default=False)
    return parser.parse_args()

@dataclass
class SequenceData:
    y_true: np.ndarray
    y_proba: np.ndarray
    position_size: np.ndarray
    X_proba: np.ndarray = None

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
            calibration_steps: int = 1,
            prob_bins: int = 50,
            disable_normalization: bool = False
        ) -> None:
        assert calibration_steps > 0, 'The number of calibration steps must be greater than 0.'
        assert prob_bins > 0, 'The number of probability bins must be greater than 0.'

        self.calibration_set_run = calibration_set_run
        self.test_set_run = test_set_run
        self.calibration_steps = calibration_steps
        self.prob_bins = prob_bins
        self.disable_normalization = disable_normalization

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
        print(f'Disable normalization: {self.disable_normalization}')
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
            "args": {
                "calibration_set_run": self.calibration_set_run.path,
                "test_set_run": self.test_set_run.path,
                "output_folder": self.output_folder_path,
                "calibration_steps": self.calibration_steps,
                "prob_bins": self.prob_bins
            },
            "calibration_set": {
                "isotonic": self.iso_stats_cal.__dict__(),
                "logistic": self.log_stats_cal.__dict__()
            },
            "test_set": {
                "isotonic": self.iso_stats_test.__dict__(),
                "logistic": self.log_stats_test.__dict__()
            }
        }
        with open(f'{self.output_folder_path}/results.yaml', 'w') as f:
            yaml.dump(results, f)
        print(f'Results saved in {self.output_folder_path}')

        self._visualize_calibration_steps()

        # Copy the info.yaml file from the calibration set to the output folder
        shutil.copy(self.calibration_set_run.run_info_path, self.output_folder_path + '/calibration_set_info.yaml')
        shutil.copy(self.test_set_run.run_info_path, self.output_folder_path + '/test_set_info.yaml')

        print('\n\nMAIN PIPELINE FINISHED.')

    def _calibration_step(self, step: int):
        """
        Convention: all the logs written from a calibration step
        should have -- at the beginning of the line.
        """
        num_logits = int(len(self.cal_proba.X_proba) * (step) / args.calibration_steps)
        print(f'-- Calibration set size: {num_logits} logits/probabilities.')

        step_folder_path = f'{self.output_folder_path}/step_{step}-{args.calibration_steps}'

        # Effective calibration data for the calibration step i
        X_proba_cal = self.cal_proba.X_proba[:num_logits]
        y_true_cal = self.cal_proba.y_true[:num_logits]

        # Isotonic regression
        iso_regressor = None
        if step == 0:
            os.makedirs(step_folder_path, exist_ok=True)
            ppl_cal, brier_score_cal, ppl_test, brier_score_test = self._evaluate(
                step,
                num_logits,
                self.iso_stats_cal,
                self.iso_stats_test,
                step_folder_path,
                regressor=None)
            self.log_stats_cal.add(num_logits, ppl_cal, brier_score_cal)
            self.log_stats_test.add(num_logits, ppl_test, brier_score_test)
        else:
            # Calibrate with isotonic regression
            min_prob = np.min(X_proba_cal) # Set minimum value to avoid division by 0 when calculating perplexity
            iso_regressor = IsotonicRegression(out_of_bounds='clip', y_min=min_prob, y_max=1)
            iso_regressor.fit(X_proba_cal, y_true_cal)
            print(f'-- Isotonic Regressor has been trained. Starting calibration...')

            iso_regressor_folder_path = f'{step_folder_path}/isotonic_regressor'
            os.makedirs(iso_regressor_folder_path, exist_ok=True)
            joblib.dump(iso_regressor, f'{iso_regressor_folder_path}/model.joblib')
            print(f'-- Model saved.')
            self._evaluate(
                step,
                num_logits,
                self.iso_stats_cal,
                self.iso_stats_test,
                iso_regressor_folder_path,
                regressor=iso_regressor)
            
            # Calibrate with logistic regression
            log_regressor = LogisticRegression(solver='lbfgs')
            # NOTE: calibration is done on the logits
            X_logits_cal = inverse_sigmoid(X_proba_cal)
            log_regressor.fit(X_logits_cal, y_true_cal)
            print(f'-- Logistic Regressor has been trained. Starting calibration...')

            log_regressor_folder_path = f'{step_folder_path}/logistic_regressor'
            os.makedirs(log_regressor_folder_path, exist_ok=True)
            joblib.dump(log_regressor, f'{log_regressor_folder_path}/model.joblib')
            print(f'-- Model saved.')
            self._evaluate(
                step,
                num_logits,
                self.log_stats_cal,
                self.log_stats_test,
                log_regressor_folder_path,
                regressor=log_regressor)

        print(f'-- Evaluation completed')
        print(f'-- Calibration step {step} completed.')

    def _calibrate(self, regressor, seq_data: SequenceData):
        if isinstance(regressor, IsotonicRegression):
            X_proba = regressor.transform(seq_data.X_proba)
            if self.disable_normalization:
                return X_proba
            else:
                return normalize(X_proba, seq_data.position_size)
        elif isinstance(regressor, LogisticRegression):
            X_logits = inverse_sigmoid(seq_data.X_proba)
            X_proba = regressor.predict_proba(X_logits)[:, 1]
            if self.disable_normalization:
                return X_proba
            else:
                return normalize(X_proba, seq_data.position_size)
        else:
            raise ValueError('Unknown regressor type.')
        
    def checkSorted(self, y_value, position_size):
        for i in range(0, len(y_value), position_size):
            to_test = y_value[i+1:i+position_size] # Skip the first element because it is the correct token
            first = to_test[0]
            for j in range(1, len(to_test)):
                if first >= to_test[j]:
                    first = to_test[j]
                else:
                    return False
        return True

    def _evaluate(
            self, 
            step: int, 
            num_logits: int,
            stats_cal: CalibrationStepStats,
            stats_test: CalibrationStepStats,
            save_folder_path: str,
            regressor = None
        ):
        # Calibration set
        save_path_cal = f'{save_folder_path}/calibration_set_'
        y_proba_cal = self.cal_proba.y_proba
        if step > 0:
            y_proba_cal = self._calibrate(regressor, self.cal_proba)
        ppl_cal, brier_score_cal = evaluate(
            self.cal_proba.y_true, 
            y_proba_cal, 
            save_path_cal, 
            prob_bins=self.prob_bins)
        stats_cal.add(num_logits, ppl_cal, brier_score_cal)

        # Test set
        save_path_test = f'{save_folder_path}/test_set_'
        y_proba_test = self.test_proba.y_proba
        if step > 0:
            y_proba_test = self._calibrate(regressor, self.test_proba)

            print("\n\n\n")
            print(
                "checkSorted(y_proba_test, self.test_proba.y_proba): ", 
                self.checkSorted(self.test_proba.y_proba, self.test_proba.position_size))


            print(
                "checkSorted(y_proba_test, self.test_proba.position_size): ", 
                self.checkSorted(y_proba_test, self.test_proba.position_size))
            



        ppl_test, brier_score_test = evaluate(
            self.test_proba.y_true, 
            y_proba_test, 
            save_path_test, 
            prob_bins=self.prob_bins)
        stats_test.add(num_logits, ppl_test, brier_score_test)

        return ppl_cal, brier_score_cal, ppl_test, brier_score_test

    def _load_data(self, run_info: RunInfo) -> SequenceData:
        position_result_proba = [
                logits_to_proba(logits) 
                for logits in run_info.logits_reader.read()
            ]
        
        y_true, y_proba, position_size = position_result_to_numpy(position_result_proba)
        data = SequenceData(y_true, y_proba, position_size)
        data.X_proba = y_proba.reshape(-1, 1)
        return data
    
    def _visualize_calibration_steps(self):
        folder_path = f'{self.output_folder_path}/general'
        folder_path_over_steps = f'{folder_path}/over_steps'
        folder_path_over_num_logits = f'{folder_path}/over_num_logits'
        os.makedirs(folder_path_over_steps, exist_ok=True)
        os.makedirs(folder_path_over_num_logits, exist_ok=True)

        # Over steps
        visualize_changes(
            range(self.calibration_steps + 1),
            [self.iso_stats_cal.ppl, self.log_stats_cal.ppl],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Calibration steps',
            ylabel='Perplexity',
            save_name=f'{folder_path_over_steps}/ppl_cal.png'
        )
        visualize_changes(
            range(self.calibration_steps + 1),
            [self.iso_stats_test.ppl, self.log_stats_test.ppl],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Calibration steps',
            ylabel='Perplexity',
            save_name=f'{folder_path_over_steps}/ppl_test.png'
        )
        visualize_changes(
            range(self.calibration_steps + 1),
            [self.iso_stats_cal.brier_score, self.log_stats_cal.brier_score],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Calibration steps',
            ylabel='Brier score',
            save_name=f'{folder_path_over_steps}/brier_score_cal.png'
        )
        visualize_changes(
            range(self.calibration_steps + 1),
            [self.iso_stats_test.brier_score, self.log_stats_test.brier_score],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Calibration steps',
            ylabel='Brier score',
            save_name=f'{folder_path_over_steps}/brier_score_test.png'
        )

        # Over num_logits
        visualize_changes(
            self.iso_stats_cal.num_logits,
            [self.iso_stats_cal.ppl, self.log_stats_cal.ppl],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Number of logits',
            ylabel='Perplexity',
            save_name=f'{folder_path_over_num_logits}/ppl_cal.png'
        )
        visualize_changes(
            self.iso_stats_test.num_logits,
            [self.iso_stats_test.ppl, self.log_stats_test.ppl],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Number of logits',
            ylabel='Perplexity',
            save_name=f'{folder_path_over_num_logits}/ppl_test.png'
        )
        visualize_changes(
            self.iso_stats_cal.num_logits,
            [self.iso_stats_cal.brier_score, self.log_stats_cal.brier_score],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Number of logits',
            ylabel='Brier score',
            save_name=f'{folder_path_over_num_logits}/brier_score_cal.png'
        )
        visualize_changes(
            self.iso_stats_test.num_logits,
            [self.iso_stats_test.brier_score, self.log_stats_test.brier_score],
            ['Isotonic regression', 'Platt scaling'],
            xlabel='Number of logits',
            ylabel='Brier score',
            save_name=f'{folder_path_over_num_logits}/brier_score_test.png'
        )
        
        



if __name__ == "__main__":
    print('Reading arguments...')

    args = read_args()

    pipeline = MainPipeline(
        RunInfo(args.calibration_set_run),
        RunInfo(args.test_set_run),
        args.output_folder,
        args.calibration_steps,
        prob_bins=args.prob_bins
    )
    pipeline.run()
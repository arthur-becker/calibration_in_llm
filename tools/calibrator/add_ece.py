import argparse
import os
import yaml

CALIBRATOR_OUTPUTS_DIR = "./../../outputs/calibrator"
EXTRACTOR_OUTPUTS_DIR = "./../../outputs/extractor"

CALIBRATION_SET_INFO_FILENAME = "calibration_set_info.yaml"
TEST_SET_INFO_FILENAME = "test_set_info.yaml"
RESULTS_FILENAME = "results.yaml"

def apply_to_calibrator_type(calibrator_type, run_path, cal_steps_list):
    for cal_step in cal_steps_list:
        step = cal_step["calibration_step"]
        cal_step["I WAS HERE"] = True
        print(f"Processing {calibrator_type} step {cal_step}")
        break

def apply_to_run(run_path, run_info):
    isotonic_steps = run_info["isotonic"]
    logistic_steps = run_info["logistic"]

    if isotonic_steps is not None:
        apply_to_calibrator_type("isotonic", run_path, isotonic_steps)
        print
    
    if logistic_steps is not None:
        apply_to_calibrator_type("logistic", run_path, logistic_steps)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add ECE to the results")
    parser.add_argument("output_dir", type=str, help="The directory containing the results of the calibrator")
    args = parser.parse_args()

    output_dir_path = CALIBRATOR_OUTPUTS_DIR + "/" + args.output_dir
    if not os.path.exists(output_dir_path):
        print(f"Directory {output_dir_path} does not exist")
        exit(1)

    print(f"Adding ECE to the results in {args.output_dir}")

    # Load the results
    with open(output_dir_path + "/" + RESULTS_FILENAME, "r") as f:
        results = yaml.safe_load(f)

        test_run_path = results["test_set"]["path"]
        calibration_run_path = results["calibration_set"]["path"]

        if not os.path.exists(test_run_path) or not os.path.exists(calibration_run_path):
            print("Paths to test and calibration runs are not valid")
            exit(1)

        print("Paths to test and calibration runs are valid")

        calibration_run_info = results["calibration_set"]
        test_set_info = results["test_set"]

        apply_to_run(calibration_run_path, calibration_run_info)
        apply_to_run(test_run_path, test_set_info)




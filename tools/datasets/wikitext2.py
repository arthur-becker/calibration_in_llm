
# SOURCE: https://paperswithcode.com/dataset/wikitext-2

import os
import json
import requests
import shutil

DATASET_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

DATASETS_OUTPUT_PATH = "./../../outputs/datasets"

TMP_ZIP_PATH = DATASETS_OUTPUT_PATH+"/wikitext2.zip"
TMP_UNZIP_PATH = DATASETS_OUTPUT_PATH+"/wikitext-2-raw"
TMP_TRAIN_FILEPATH = TMP_UNZIP_PATH+"/wiki.train.raw"
TMP_TEST_FILEPATH = TMP_UNZIP_PATH+"/wiki.test.raw"

TARGET_FILESIZE = 1_300_000
TARGET_TRAIN_FILEPATH = DATASETS_OUTPUT_PATH+"/wikitext2_train.txt"
TARGET_TEST_FILEPATH = DATASETS_OUTPUT_PATH+"/wikitext2_test.txt"

def download_file(url, filename):
    response = requests.get(url, allow_redirects=True)
    with open(filename, "wb") as file:
        file.write(response.content)

def copy_lines_fixed_size(source_file, target_filepath, target_filesize):
    filesize = 0

    result = ""
    i = 0
    while filesize < target_filesize:

        line = source_file.readline()
        result += line

        filesize += len(line)
        print(f"Filesize: {filesize} / {target_filesize}")
        i = i + 1

    with open(target_filepath, "w") as f:
        f.write(result)

if __name__ == "__main__":

    print("Downloading wikitext2 dataset...")

    # Download the dataset
    if os.path.exists(TMP_ZIP_PATH):
        print("File already exists. Skipping download...")
    else:
        download_file(DATASET_URL, TMP_ZIP_PATH)

    # Unzip the dataset
    os.system(f"unzip {TMP_ZIP_PATH} -d {DATASETS_OUTPUT_PATH}")

    # Remove the downloaded file
    os.remove(TMP_ZIP_PATH)

    # Create the train and test datasets
    with open(TMP_TRAIN_FILEPATH, "r") as f:
        print("Creating train dataset...")
        copy_lines_fixed_size(f, TARGET_TRAIN_FILEPATH, TARGET_FILESIZE)
    with open(TMP_TEST_FILEPATH, "r") as f:
        print("Creating test dataset...")
        with open(TARGET_TEST_FILEPATH, "w") as f2:
            f2.write(f.read())

    # Remove the unzipped files
    shutil.rmtree(TMP_UNZIP_PATH)

    print("Done!")
    
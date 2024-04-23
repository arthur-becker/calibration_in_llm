import json
import requests
import os

# Source: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample
URL = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/arxiv_sample.jsonl"

LOCAL_TMP_FILEPATH = "./../../outputs/datasets/redpajama_arxiv.jsonl"
TARGET_FILESIZE = 1_000_000

TARGET_FILEPATH_TRAIN = "./../../outputs/datasets/redpajama_arxiv_train.txt"
TARGET_FILEPATH_TEST = "./../../outputs/datasets/redpajama_arxiv_test.txt"

def download_file(url, filename):
    response = requests.get(url, allow_redirects=True)
    with open(filename, "wb") as file:
        file.write(response.content)

def create_dataset_from_jsonl(source_file, target_filepath, target_filesize):
    filesize = 0

    result = ""
    i = 0
    while filesize < target_filesize:

        l = source_file.readline()
        json_obj = json.loads(l)
        text = json_obj["text"]
        result += text + "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"

        filesize += len(text)
        print(f"Filesize: {filesize} / {target_filesize}")
        i = i + 1

    with open(target_filepath, "w") as f:
        f.write(result)

if __name__ == "__main__":

    print("Downloading RedPajama-Arxiv dataset...")
    
    if os.path.exists(LOCAL_TMP_FILEPATH):
        print("File already exists. Skipping download...")
    else:
        download_file(URL, LOCAL_TMP_FILEPATH)

    with open(LOCAL_TMP_FILEPATH, "r") as f:

        print("Creating train dataset...")
        create_dataset_from_jsonl(f, TARGET_FILEPATH_TRAIN, TARGET_FILESIZE)

        print("Creating test dataset...")
        create_dataset_from_jsonl(f, TARGET_FILEPATH_TEST, TARGET_FILESIZE)

    # Remove the downloaded file
    os.remove(LOCAL_TMP_FILEPATH)
    
    print("Done!")
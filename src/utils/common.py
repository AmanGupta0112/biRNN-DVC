import os
import yaml
import time
import pandas as pd
import json
import joblib
from src import logging

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")

def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


def save_bin(data, path):
    joblib.dump(value=data,filename=path)
    logging.info(f"Binary file saved at: {path}")


def load_bin(path):
    data = joblib.load(path)
    logging.info(f"Binary file loaded from: {path}")
    return data

def uniquename():
    return time.strftime("%Y%m%d%H%M%S")
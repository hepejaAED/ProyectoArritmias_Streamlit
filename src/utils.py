import json
import os

THRESHOLD_PATH = "models/threshold.json"

def load_threshold(default=0.51):
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            return json.load(f).get("threshold", default)
    return default


def save_threshold(value):
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": float(value)}, f)
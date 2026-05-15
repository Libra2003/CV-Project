import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def env_path(key):
    value = os.getenv(key)
    if value is None:
        raise RuntimeError(f"Missing environment variable: {key}")
    return Path(value)


DATASET_ROOT = env_path("DATASET_ROOT")
EDGE_MAPS_ROOT = env_path("EDGE_MAPS_ROOT")
LOG_ROOT = env_path("LOG_ROOT")
CHECKPOINT_ROOT = env_path("CHECKPOINT_ROOT")
MANO_MODELS_ROOT = env_path("MANO_MODELS_ROOT")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
GDRIVE_CREDENTIALS_PATH = env_path("GDRIVE_CREDENTIALS_PATH")
GDRIVE_TOKEN_PATH = env_path("GDRIVE_TOKEN_PATH")

import os
import urllib.request
import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = Path(os.getenv("DATASET_ROOT"))
EDGE_MAPS_ROOT = Path(os.getenv("EDGE_MAPS_ROOT"))
MODEL_PATH = Path("hand_landmarker.task")
MODEL_DOWNLOAD_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_model_exists():
    if MODEL_PATH.exists():
        return
    print("Hand landmarker model not found, downloading...")
    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_PATH)
    print("Model downloaded successfully")


def create_hand_mask(image, hand_landmarks_list):
    image_height, image_width = image.shape[:2]
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for hand_landmarks in hand_landmarks_list:
        landmark_points = np.array([
            (int(landmark.x * image_width), int(landmark.y * image_height))
            for landmark in hand_landmarks
        ])
        convex_hull = cv2.convexHull(landmark_points)
        cv2.fillConvexPoly(mask, convex_hull, 255)

    return mask


def generate_edge_map(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    grayscale_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(grayscale_image, threshold1=50, threshold2=150)


def build_output_path(image_path):
    sequence_name = image_path.parent.parent.name
    output_path = EDGE_MAPS_ROOT / sequence_name / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def process_dataset():
    ensure_model_exists()

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_hands=2
        )
    )

    image_paths = sorted(DATASET_ROOT.rglob("rgb/*.jpg"))
    skipped_count = 0
    processed_count = 0

    for image_path in tqdm(image_paths, desc="Generating edge maps", unit="img"):
        mp_image = mp.Image.create_from_file(str(image_path))
        detection_results = detector.detect(mp_image)

        if not detection_results.hand_landmarks:
            skipped_count += 1
            continue

        image = cv2.imread(str(image_path))
        hand_mask = create_hand_mask(image, detection_results.hand_landmarks)
        edge_map = generate_edge_map(image, hand_mask)

        cv2.imwrite(str(build_output_path(image_path)), edge_map)
        processed_count += 1

    print(f"\nDone! Processed: {processed_count} | Skipped: {skipped_count}")


process_dataset()

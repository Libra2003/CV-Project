import re
import pickle
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from paths import DATASET_ROOT, EDGE_MAPS_ROOT


SUBJECT_SEQUENCE_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)$")
HELDOUT_SEQUENCES_PER_SUBJECT = 1
INPUT_RESOLUTION = 256
RECORDS_CACHE_FILENAME = "records_cache.pt"


def parse_sequence_name(sequence_name):
    match = SUBJECT_SEQUENCE_PATTERN.match(sequence_name)
    if not match:
        raise ValueError(f"Cannot parse sequence name: {sequence_name}")
    return match.group(1), int(match.group(2))


def group_sequences_by_subject(sequence_names):
    groups = {}
    for name in sequence_names:
        subject, _ = parse_sequence_name(name)
        groups.setdefault(subject, []).append(name)
    for subject in groups:
        groups[subject].sort(key=lambda name: parse_sequence_name(name)[1])
    return groups


def split_sequences(all_sequences, heldout_per_subject):
    grouped = group_sequences_by_subject(all_sequences)
    train_sequences = []
    val_sequences = []
    for sequences in grouped.values():
        num_val = min(heldout_per_subject, max(0, len(sequences) - 2))
        if num_val == 0:
            train_sequences.extend(sequences)
            continue
        train_sequences.extend(sequences[:-num_val])
        val_sequences.extend(sequences[-num_val:])
    return sorted(train_sequences), sorted(val_sequences)


def discover_sequences(dataset_root):
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def resolve_meta_dir(sequence_dir):
    release_dir = sequence_dir / "meta_Release"
    if release_dir.is_dir():
        return release_dir
    return sequence_dir / "meta"


def collect_frame_records(dataset_root, edge_maps_root, sequences):
    from tqdm import tqdm
    records = []
    for sequence_name in tqdm(sequences, desc="Scanning sequences"):
        sequence_dir = dataset_root / sequence_name
        meta_dir = resolve_meta_dir(sequence_dir)
        rgb_dir = sequence_dir / "rgb"
        edge_dir = edge_maps_root / sequence_name

        for rgb_path in sorted(rgb_dir.glob("*.jpg")):
            frame_id = rgb_path.stem
            edge_path = edge_dir / f"{frame_id}.jpg"
            meta_path = meta_dir / f"{frame_id}.pkl"
            if not edge_path.exists() or not meta_path.exists():
                continue
            if not has_required_annotations(meta_path):
                continue
            records.append({
                "sequence": sequence_name,
                "frame_id": frame_id,
                "edge_path": edge_path,
                "meta_path": meta_path,
            })
    return records


def has_required_annotations(meta_path):
    meta = load_meta_annotation(meta_path)
    required_keys = ["camMat", "handPose", "handBeta", "handJoints3D", "handTrans"]
    return all(meta.get(key) is not None for key in required_keys)


def load_meta_annotation(meta_path):
    with open(meta_path, "rb") as handle:
        return pickle.load(handle, encoding="latin1")


def compute_bbox_from_edge_map(edge_map, padding):
    nonzero_y, nonzero_x = np.nonzero(edge_map)
    if nonzero_y.size == 0:
        height, width = edge_map.shape
        return 0, 0, width, height
    min_x, max_x = nonzero_x.min(), nonzero_x.max()
    min_y, max_y = nonzero_y.min(), nonzero_y.max()
    height, width = edge_map.shape
    x0 = max(0, min_x - padding)
    y0 = max(0, min_y - padding)
    x1 = min(width, max_x + padding)
    y1 = min(height, max_y + padding)
    return x0, y0, x1, y1


def crop_and_resize(image, bbox, target_size):
    x0, y0, x1, y1 = bbox
    cropped = image[y0:y1, x0:x1]
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)


def adjust_intrinsics_for_crop(camera_matrix, bbox, target_size):
    x0, y0, x1, y1 = bbox
    crop_width = x1 - x0
    crop_height = y1 - y0
    scale_x = target_size / crop_width
    scale_y = target_size / crop_height

    adjusted = camera_matrix.copy().astype(np.float32)
    adjusted[0, 0] *= scale_x
    adjusted[1, 1] *= scale_y
    adjusted[0, 2] = (camera_matrix[0, 2] - x0) * scale_x
    adjusted[1, 2] = (camera_matrix[1, 2] - y0) * scale_y
    return adjusted


class HO3DDataset(Dataset):
    def __init__(self, split, bbox_padding=20, target_size=INPUT_RESOLUTION):
        self.bbox_padding = bbox_padding
        self.target_size = target_size
        self.records = self.load_or_build_records(split)

    def load_or_build_records(self, split):
        cache_path = EDGE_MAPS_ROOT / f"{split}_{RECORDS_CACHE_FILENAME}"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=False)

        all_sequences = discover_sequences(DATASET_ROOT)
        train_sequences, val_sequences = split_sequences(all_sequences, HELDOUT_SEQUENCES_PER_SUBJECT)
        selected_sequences = train_sequences if split == "train" else val_sequences
        records = collect_frame_records(DATASET_ROOT, EDGE_MAPS_ROOT, selected_sequences)
        torch.save(records, cache_path)
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        edge_map = cv2.imread(str(record["edge_path"]), cv2.IMREAD_GRAYSCALE)
        meta = load_meta_annotation(record["meta_path"])

        bbox = compute_bbox_from_edge_map(edge_map, self.bbox_padding)
        edge_cropped = crop_and_resize(edge_map, bbox, self.target_size)
        camera_matrix = adjust_intrinsics_for_crop(meta["camMat"], bbox, self.target_size)

        edge_tensor = torch.from_numpy(edge_cropped).float().unsqueeze(0) / 255.0
        theta = torch.from_numpy(np.asarray(meta["handPose"], dtype=np.float32))
        beta = torch.from_numpy(np.asarray(meta["handBeta"], dtype=np.float32))
        joints_3d = torch.from_numpy(np.asarray(meta["handJoints3D"], dtype=np.float32))
        translation = torch.from_numpy(np.asarray(meta["handTrans"], dtype=np.float32))
        camera_tensor = torch.from_numpy(camera_matrix)

        return {
            "edge_map": edge_tensor,
            "theta": theta,
            "beta": beta,
            "joints_3d": joints_3d,
            "translation": translation,
            "camera_matrix": camera_tensor,
            "sequence": record["sequence"],
            "frame_id": record["frame_id"],
        }

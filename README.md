# CVTAH: Mobile Hand Mesh Reconstruction

Lightweight 3D hand mesh recovery from edge maps via sparse convolutions.

## Project Overview

A small (~1.5M param) sparse convolutional model that predicts 3D hand mesh (MANO parameters → vertices and joints) from edge maps extracted via MediaPipe + Canny. The "mobile" angle: instead of running dense convolutions over RGB pixels (most of which are background), we extract hand edges first and run a sparse network only on edge coordinates.

**Trained on:** HO-3D v3 with ground-truth MANO supervision.

---

## Pipeline

1. **Edge map generation** (`generate_edge_maps.py`) — MediaPipe locates hands, convex hull masks the ROI, Canny extracts edges. Run once, output saved to disk.
2. **Training** (`train.py`) — sparse student model + MANO layer + multi-term loss (param L1 + vertex MSE + joint MSE).
3. **Logging** — TensorBoard event files written locally and synced to Google Drive in the background.

---

## Setup

### 1. Create virtual environment (Python 3.12)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install PyTorch with CUDA

Match your CUDA version (the example below is for CUDA 12.4):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install remaining dependencies

```powershell
pip install mediapipe opencv-contrib-python numpy tqdm python-dotenv tensorboard watchdog google-auth google-auth-oauthlib google-api-python-client
pip install spconv-cu124
pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git
pip install git+https://github.com/hassony2/manopth.git
```

### 4. MANO model weights

Register at https://mano.is.tue.mpg.de, download the model archive, and place `MANO_RIGHT.pkl` at `mano_models/MANO_RIGHT.pkl` (or whatever path you set in `MANO_MODELS_ROOT`).

### 5. Google Drive log sync (optional)

Follow Google Cloud Console steps to enable the Drive API, create a Desktop OAuth client, and download `credentials.json` to the project root. Add yourself as a test user on the OAuth consent screen. Create a destination folder on Drive and copy its folder ID.

### 6. Configure `.env`

```
DATASET_ROOT=C:\path\to\HO3D_v3\train
EDGE_MAPS_ROOT=D:\Projects\CVTAH\data\edge_maps
LOG_ROOT=D:\Projects\CVTAH\runs
CHECKPOINT_ROOT=D:\Projects\CVTAH\checkpoints
MANO_MODELS_ROOT=D:\Projects\CVTAH\mano_models
MODEL_PATH=D:\Projects\CVTAH\hand_landmarker.task
GDRIVE_FOLDER_ID=your_folder_id_here
GDRIVE_CREDENTIALS_PATH=D:\Projects\CVTAH\credentials.json
GDRIVE_TOKEN_PATH=D:\Projects\CVTAH\token.json
```

---

## Stage 1: Generate edge maps

One-time preprocessing of the HO-3D dataset:

```powershell
python generate_edge_maps.py
```

The script auto-downloads the MediaPipe hand landmarker on first run.

---

## Stage 2: Train

```powershell
python train.py --run-name v0_baseline --batch-size 32 --num-workers 4 --num-epochs 100
```

Monitor with TensorBoard:

```powershell
tensorboard --logdir runs
```

Or check the synced folder on Google Drive.

**Resume an interrupted run** by re-running the same command — checkpoints are loaded automatically from `checkpoints/<run_name>/latest.pt`.

---

## Architecture Summary

- **Input:** edge map, 256×256, single channel (cropped to hand bbox)
- **Backbone:** sparse ConvNeXt-style — stem (1→32) + 3 stages with stride-2 downsampling (32→64→128→256)
- **Pool:** global mean pooling over active sparse cells
- **Head:** 3-layer MLP, 512 hidden, → 58-dim output (48 θ + 10 β)
- **MANO layer:** frozen `manopth` with HO-3D coordinate alignment baked in
- **Losses:** param L1 (θ + β) + vertex MSE + joint MSE

---

## Project Structure

.
├── paths.py # env-driven path resolution
├── ho3d_dataset.py # HO-3D loader, sequence-level split, cached records
├── mano_wrapper.py # MANO layer + HO-3D coordinate alignment
├── losses.py # multi-term hand reconstruction loss
├── student_model.py # sparse ConvNeXt + MLP head
├── checkpointing.py # atomic save/load with full training state
├── gdrive_log_sync.py # debounced background Drive uploader
├── train.py # training loop
└── generate_edge_maps.py # offline edge map preprocessing

---

## Authors

- Tayyab Zain Up Abideen
- Ahmed Saleem

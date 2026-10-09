# pecan-live-yolo

Live camera + YOLO (or cascade U-Net) overlay for pecan videos.

Camera backends: **Teledyne Spinnaker** (PySpin), OpenCV webcam, or a video file. This repo is intentionally lean — it does **not** install the full napari plugin stack.

## Setup (Windows + Spinnaker)

1. Install Teledyne Spinnaker SDK (includes the `cp310` PySpin wheel).
2. From this repo root:

```powershell
powershell -File scripts/setup_spinnaker_env.ps1
```

That creates `.venv-spinnaker` with Python 3.10, CUDA PyTorch, ultralytics, OpenCV GUI, and the Spinnaker wheel.

3. Copy checkpoints into the layout expected by the script, or set env vars:

```text
training_runs/yolo_seg_runs/<run>/<run>.pt
training_runs/cascade_seg_runs/<name>.pt
```

Env overrides: `PECAN_YOLO_WEIGHTS`, `PECAN_CASCADE_WEIGHTS`, `PECAN_YOLO_RUN`.

## Run

```bat
scripts\run_live_yolo.bat
```

or:

```bat
.venv-spinnaker\Scripts\python.exe scripts\live_yolo.py
.venv-spinnaker\Scripts\python.exe scripts\live_yolo.py --video path.mp4
```

Notebook: open `live_yolo_inference.ipynb` with the **Pecan PySpin (Python 3.10)** kernel.

## Cascade backend

Pure YOLO mode needs only this env. Cascade mode imports `pecanpy.widgets.cascade_seg` from **napari-pecan-py**:

```bash
# in .venv-spinnaker (or set PECAN_PLUGIN_SRC to that repo's src/)
uv pip install --python .venv-spinnaker\Scripts\python.exe -e ..\napari-pecan-py
```

Sibling checkouts named `napari-pecan-py` or `pecan_py_napari` are also searched automatically.

## Layout

```text
scripts/live_yolo.py           # main live loop
scripts/setup_spinnaker_env.ps1
scripts/run_live_yolo.bat
live_yolo_inference.ipynb
weights/                       # optional local .pt copies
```

Extracted from [pecan_py_napari](https://github.com/salirezav/pecan_py_napari) (napari-pecan-py).

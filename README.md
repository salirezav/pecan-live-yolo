# pecan-live-yolo

Live camera + YOLO (or cascade U-Net) overlay for pecan videos.

Default camera: **ELP / Ailipu global shutter** over OpenCV DirectShow (UVC). Also supports Teledyne Spinnaker (PySpin) or a video file. This repo is intentionally lean — it does **not** install the full napari plugin stack.

## Setup (Windows + uv)

From this repo root:

```powershell
uv sync
```

That creates `.venv` with Python 3.12, CUDA PyTorch, ultralytics, OpenCV GUI, and the other live-inference packages.

Checkpoints live under `weights/` (preferred) or the plugin `training_runs/` layout:

```text
weights/2026-08-19_131846.pt   # default pecan/kernel demo
weights/halfyield-model.pt     # half / piece / shell / obscured
```

`scripts\run_live_yolo.bat` lists `weights/*.pt` and asks which model to load. Or skip the picker:

```bat
scripts\run_live_yolo.bat --weights halfyield-model.pt
scripts\run_live_yolo.bat --list-weights
```

Env overrides still work: `PECAN_YOLO_WEIGHTS`, `PECAN_CASCADE_WEIGHTS`, `PECAN_YOLO_RUN`.

### Optional Spinnaker (Blackfly)

Teledyne only ships a **cp310** PySpin wheel. For GigE/USB Blackfly cameras:

```powershell
powershell -File scripts/setup_spinnaker_env.ps1
```

Then set `CAMERA_BACKEND = "spinnaker"` in the script or notebook.

## Run

```bat
scripts\run_live_yolo.bat
```

or:

```bat
.venv\Scripts\python.exe scripts\live_yolo.py
.venv\Scripts\python.exe scripts\live_yolo.py --video path.mp4
```

Notebook: open `live_yolo_inference.ipynb` with the **`.venv` (Python 3.12)** kernel (`Pecan Live YOLO`).

## Cascade backend

Pure YOLO mode needs only this env. Cascade mode imports `pecanpy.widgets.cascade_seg` from **napari-pecan-py**:

```bash
uv pip install --python .venv\Scripts\python.exe -e ..\napari-pecan-py
```

Sibling checkouts named `napari-pecan-py` or `pecan_py_napari` are also searched automatically.

## Layout

```text
scripts/live_yolo.py           # main live loop
scripts/run_live_yolo.bat
scripts/setup_spinnaker_env.ps1  # optional Blackfly env
live_yolo_inference.ipynb
weights/                       # optional local .pt copies
pyproject.toml
```

Extracted from [pecan_py_napari](https://github.com/salirezav/pecan_py_napari) (napari-pecan-py).

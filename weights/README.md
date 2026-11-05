# Weights

Local YOLO-seg / cascade checkpoints used by `scripts/live_yolo.py`.

| File | Source | Notes |
|---|---|---|
| `2026-08-19_131846.pt` | `pecan_py_napari/training_runs/yolo_seg_runs/...` | Default live demo YOLO-seg (pecan / kernel) |
| `demo-yolo-seg.pt` | same file | Alias of the demo checkpoint |
| `halfyield-model.pt` | half-yield-analysis zip (`halfyield-model.pt`) | YOLOv12n-seg: half / piece / shell / obscured |

## Switching models

Double-click / run `scripts\run_live_yolo.bat` — it lists `weights/*.pt` and asks which to load.

```bat
scripts\run_live_yolo.bat
scripts\run_live_yolo.bat --weights halfyield-model.pt
scripts\run_live_yolo.bat --list-weights
```

Notebook / Python:

```python
live.set_weights("halfyield-model.pt")
# live.set_weights("2026-08-19_131846")  # demo pecan/kernel
```

Also accepted: `PECAN_YOLO_WEIGHTS`, `PECAN_YOLO_RUN`, `PECAN_CASCADE_WEIGHTS`.

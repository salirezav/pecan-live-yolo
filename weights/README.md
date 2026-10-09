# Weights

Copy YOLO-seg or cascade checkpoints here (or keep the plugin `training_runs/` layout).

Default YOLO path (relative to repo root):

```text
training_runs/yolo_seg_runs/<run_name>/<run_name>.pt
```

Or set env vars before launch:

- `PECAN_YOLO_WEIGHTS` — path to a `.pt` YOLO-seg checkpoint
- `PECAN_CASCADE_WEIGHTS` — path to a cascade U-Net checkpoint
- `PECAN_YOLO_RUN` — run folder name under `training_runs/yolo_seg_runs/`

Cascade mode needs an installed `napari-pecan-py` (provides `pecanpy.widgets.cascade_seg`).

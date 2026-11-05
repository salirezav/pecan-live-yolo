r"""Live camera + YOLO-seg overlay (OpenCV windows).

Backends: OpenCV DirectShow (ELP UVC), Teledyne Spinnaker, or a video file.

Run with the project env:
  .venv\Scripts\python.exe scripts/live_yolo.py
Or double-click ``scripts/run_live_yolo.bat``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Windows: importing PySpin (or some OpenCV builds) first loads DLLs that make a
# later `import torch` fail with WinError 127 on shm.dll. Load torch first.
import torch

import cv2
from PIL import Image as PILImage


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        if float(np.max(img)) <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def to_yolo_predict_source(frame: np.ndarray):
    """Wrap an RGB frame for ``model.predict`` (PIL = RGB, unlike numpy = BGR)."""
    return PILImage.fromarray(_to_uint8_rgb(frame))


def inference_imgsz(height: int, width: int, model=None) -> int:
    """Pick an inference size aligned with YOLO stride (multiple of 32)."""
    if model is not None:
        for attr in ("overrides", "args"):
            args = getattr(model, attr, None)
            if isinstance(args, dict) and args.get("imgsz"):
                try:
                    saved = int(args["imgsz"])
                    if saved > 0:
                        return saved
                except (TypeError, ValueError):
                    pass
    size = max(int(height), int(width))
    return max(32, int(((size + 31) // 32) * 32))


def resolve_yolo_device(device: str) -> str:
    if str(device).strip().lower() != "auto":
        return str(device)
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
    except Exception:
        pass
    return "cpu"


# --- model ---
REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
RUNS_ROOT = REPO_ROOT / "training_runs" / "yolo_seg_runs"
# Local demo YOLO-seg checkpoint (also under weights/ as demo-yolo-seg.pt).
RUN_NAME = os.environ.get("PECAN_YOLO_RUN", "2026-08-19_131846")
# Dirty live-test switch. Flip back to "yolo" to use the YOLO-seg weights.
MODEL_BACKEND = "yolo"  # "yolo" | "cascade"
_CASCADE_LOCAL = WEIGHTS_DIR / "literally the best.pt"
_CASCADE_RUNS = (
    REPO_ROOT / "training_runs" / "cascade_seg_runs" / "literally the best.pt"
)
_CASCADE_DEFAULT = _CASCADE_LOCAL if _CASCADE_LOCAL.is_file() else _CASCADE_RUNS
CASCADE_WEIGHTS_PATH = Path(
    os.environ.get("PECAN_CASCADE_WEIGHTS", str(_CASCADE_DEFAULT))
)


def resolve_yolo_weights(
    name_or_path: str | Path | None = None,
    *,
    run_name: str | None = None,
) -> Path:
    """Resolve a YOLO ``.pt`` from env, ``weights/``, or ``training_runs/``."""
    env = os.environ.get("PECAN_YOLO_WEIGHTS")
    if env:
        return Path(env)

    raw = name_or_path
    if raw is None:
        raw = run_name or RUN_NAME
    path = Path(raw)
    if path.is_file():
        return path

    stem = path.name
    candidates = [
        WEIGHTS_DIR / stem,
        WEIGHTS_DIR / f"{stem}.pt" if not stem.endswith(".pt") else None,
        WEIGHTS_DIR / "demo-yolo-seg.pt" if str(stem).startswith("2026-08-19") else None,
        RUNS_ROOT / path.stem / f"{path.stem}.pt",
        RUNS_ROOT / stem / f"{stem}.pt" if not stem.endswith(".pt") else None,
        RUNS_ROOT / stem,
    ]
    for cand in candidates:
        if cand is not None and cand.is_file():
            return cand
    # Prefer the local weights copy in error messages when the run folder is absent.
    local = WEIGHTS_DIR / (stem if stem.endswith(".pt") else f"{stem}.pt")
    return local if not (RUNS_ROOT / path.stem / f"{path.stem}.pt").exists() else (
        RUNS_ROOT / path.stem / f"{path.stem}.pt"
    )


WEIGHTS_PATH = (
    CASCADE_WEIGHTS_PATH
    if MODEL_BACKEND == "cascade"
    else resolve_yolo_weights(RUN_NAME)
)
CONF = 0.25
DEVICE = "auto"  # auto | cpu | 0 (first CUDA GPU)

# --- overlay ---
MASK_ALPHA = 0.5
# BGR colors for OpenCV. Kernel is drawn on top of Pecan where they overlap.
CLASS_COLORS_BGR = {
    "pecan": (0, 140, 255),   # orange
    "kernel": (40, 220, 40),  # green
    "crack": (0, 40, 255),    # red (cascade U-Net only)
    # half-yield-analysis classes
    "half": (0, 200, 255),
    "piece": (40, 220, 40),
    "shell": (0, 140, 255),
    "obscured": (180, 180, 180),
}

# --- camera ---
CAMERA_BACKEND = "opencv"  # "opencv" | "spinnaker" | "video"
CAMERA_INDEX = 0
# Prefer this DirectShow / PnP friendly name (ELP AR0234). Empty = use CAMERA_INDEX.
OPENCV_CAMERA_NAME = "Global Shutter Camera"
# Default grab sizes for the ELP global shutter (MJPEG). HQ Grab toggles these.
OPENCV_FAST_SIZE = (1280, 720)
OPENCV_HQ_SIZE = (1920, 1200)
SPINNAKER_CAMERA_INDEX = 0
# Pin the GigE PoE Blackfly (BFS-PGE-23S3C). Empty string = auto (prefer GigE).
SPINNAKER_SERIAL = "26312947"
SPINNAKER_PREFER_GIGE = True
# Jumbo 9000 needs NIC jumbo frames. 0 = leave the camera default (~1400, works here).
GEV_PACKET_SIZE = 0
SPINNAKER_ROOT = Path(r"C:\Program Files\Teledyne\Spinnaker")
VIDEO_PATH = Path(r"C:\Users\salir\Desktop\demo fixture\Untitled Project.mp4")
# Connected-component pecan blobs smaller than this (px) are ignored in counts.
MIN_PECAN_AREA = 80
# Fraction of a pecan blob overlapping kernel → counted as cracked.
CRACK_OVERLAP_MIN = 0.02

# --- display / timing ---
WINDOW_NAME = "Pecan cascade live" if MODEL_BACKEND == "cascade" else "Pecan YOLO live"
CONTROLS_WINDOW_NAME = "Camera controls"
TARGET_FPS = 8  # initial display FPS; live slider can change this
# Full sensor grab + HQ demosaic/CCM, then software downscale (do NOT crop the sensor).
CAMERA_BINNING = 1          # 1 = full sensor. 2 = on-camera 2x2 (softer, faster).
DISPLAY_MAX_WIDTH = 1920
# Live "HQ Grab" switch in Camera controls (1 = quality, 0 = fast).
HQ_FRAME_MAX_WIDTH = 1920
HQ_INFER_IMGSZ = 1280
FAST_FRAME_MAX_WIDTH = 960
FAST_INFER_IMGSZ = 640
FAST_LIVE = False           # start in quality mode; toggle with HQ Grab
FRAME_MAX_WIDTH = HQ_FRAME_MAX_WIDTH
INFER_IMGSZ = HQ_INFER_IMGSZ

# --- Spinnaker color / white balance (matches SpinView "Color Correction" preset) ---
ENABLE_COLOR_CORRECTION = True
ENABLE_AUTO_WHITE_BALANCE = True  # BalanceWhiteAuto = Continuous (on-camera ISP)
CCM_APPLICATION = "GENERIC"
CCM_SENSOR = "IMX249"  # BFS-PGE-23S3C = IMX249; USB BFS-U3-51S5C = IMX250
CCM_COLOR_TEMP = "GENERAL"
CCM_COLOR_SPACE = "SRGB"  # SpinView "Color Space: ON"
CCM_TYPE = "ADVANCED"

model = None
resolved_device = "cpu"
class_names: dict[int, str] = {}
cascade_image_size = 384
cascade_torch_device = None
label_id_by_class: dict[str, int] = {}


def print_env() -> None:
    print(f"Python {sys.version.split()[0]}")
    print(f"Model backend: {MODEL_BACKEND}")
    print(f"Weights path: {WEIGHTS_PATH}")
    print(f"Weights exist: {WEIGHTS_PATH.exists()}")


def set_run(run_name: str) -> Path:
    """Point ``WEIGHTS_PATH`` at a YOLO checkpoint (``weights/`` or training run)."""
    global RUN_NAME, WEIGHTS_PATH
    RUN_NAME = str(run_name)
    WEIGHTS_PATH = resolve_yolo_weights(run_name)
    return WEIGHTS_PATH


def set_weights(name_or_path: str | Path) -> Path:
    """Point ``WEIGHTS_PATH`` at a file under ``weights/`` or an absolute ``.pt`` path.

    Examples:
      set_weights("2026-08-19_131846")
      set_weights("demo-yolo-seg.pt")
      set_weights("halfyield-model.pt")
    """
    global RUN_NAME, WEIGHTS_PATH
    WEIGHTS_PATH = resolve_yolo_weights(name_or_path)
    RUN_NAME = WEIGHTS_PATH.stem
    return WEIGHTS_PATH


def list_available_weights() -> list[Path]:
    """Sorted ``.pt`` files under ``weights/`` (non-recursive)."""
    if not WEIGHTS_DIR.is_dir():
        return []
    return sorted(
        (p for p in WEIGHTS_DIR.glob("*.pt") if p.is_file()),
        key=lambda p: p.name.lower(),
    )


def choose_weights_interactive(*, default: Path | None = None) -> Path:
    """Print available models and ask which one to load."""
    models = list_available_weights()
    if not models:
        raise FileNotFoundError(
            f"No .pt files in {WEIGHTS_DIR}. Drop checkpoints there first."
        )

    preferred = Path(default) if default is not None else Path(WEIGHTS_PATH)
    default_idx = 0
    for i, path in enumerate(models):
        if path.resolve() == preferred.resolve() or path.name == preferred.name:
            default_idx = i
            break

    print()
    print(f"Available models in {WEIGHTS_DIR}:")
    for i, path in enumerate(models, start=1):
        mb = path.stat().st_size / (1024 * 1024)
        mark = "  (default)" if (i - 1) == default_idx else ""
        print(f"  [{i}] {path.name}  ({mb:.1f} MB){mark}")
    print()

    while True:
        raw = input(f"Choose model [1-{len(models)}] (Enter={default_idx + 1}): ").strip()
        if not raw:
            choice = default_idx
            break
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            choice = int(raw) - 1
            break
        print(f"  Enter a number between 1 and {len(models)}.")

    chosen = models[choice]
    set_weights(chosen)
    print(f"Using: {chosen.name}")
    print()
    return chosen


def require_pyspin() -> None:
    """Verify Teledyne Spinnaker PySpin is available in this kernel."""
    wheel_dir = SPINNAKER_ROOT / "PySpin"
    wheels = sorted(wheel_dir.glob("spinnaker_python-*.whl"))
    wheel_tags = [w.name.split("-")[4] for w in wheels if len(w.name.split("-")) > 4]
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    try:
        import PySpin

        if not hasattr(PySpin, "System"):
            raise ImportError("wrong PySpin package (PyPI pyspin vs Spinnaker SDK)")
        ver = PySpin.System.GetInstance().GetLibraryVersion()
        print(f"PySpin OK — Spinnaker {ver.major}.{ver.minor}.{ver.type}.{ver.build} (Python {py_ver})")
    except ImportError as exc:
        raise RuntimeError(
            f"PySpin not available (Python {py_ver}).\n"
            f"Spinnaker wheels on disk: {wheel_tags or ['none']}\n\n"
            "Switch kernel to 'Pecan PySpin (Python 3.10)' or run:\n"
            "  powershell -File scripts/setup_spinnaker_env.ps1\n\n"
            "Or set CAMERA_BACKEND = 'opencv' for a generic webcam."
        ) from exc


_CCM_APPLICATION = {
    "GENERIC": "SPINNAKER_CCM_APPLICATION_GENERIC",
    "MICROSCOPY": "SPINNAKER_CCM_APPLICATION_MICROSCOPY",
}
_CCM_SENSOR = {
    "IMX249": "SPINNAKER_CCM_SENSOR_IMX249",
    "IMX250": "SPINNAKER_CCM_SENSOR_IMX250",
}
_CCM_SENSOR_BY_MODEL = {
    "BFS-PGE-23S3C": "IMX249",
    "BFS-U3-51S5C": "IMX250",
}
_CCM_COLOR_TEMP = {
    "GENERAL": "SPINNAKER_CCM_COLOR_TEMP_GENERAL",
    "SUNNY_5000K": "SPINNAKER_CCM_COLOR_TEMP_SUNNY_5000K",
    "CLOUDY_6500K": "SPINNAKER_CCM_COLOR_TEMP_CLOUDY_6500K",
    "TUNGSTEN_2800K": "SPINNAKER_CCM_COLOR_TEMP_TUNGSTEN_2800K",
    "DAYLIGHT_5034K": "SPINNAKER_CCM_COLOR_TEMP_DAYLIGHT_5034K",
}
_CCM_COLOR_SPACE = {
    "SRGB": "SPINNAKER_CCM_COLOR_SPACE_SRGB",
    "OFF": "SPINNAKER_CCM_COLOR_SPACE_OFF",
}
_CCM_TYPE = {
    "ADVANCED": "SPINNAKER_CCM_TYPE_ADVANCED",
    "LINEAR": "SPINNAKER_CCM_TYPE_LINEAR",
}


def _pyspin_enum_value(pyspin_module, name: str):
    return getattr(pyspin_module, name)


def _set_spinnaker_enum(nodemap, pyspin_module, node_name: str, entry_name: str) -> bool:
    node = pyspin_module.CEnumerationPtr(nodemap.GetNode(node_name))
    if not pyspin_module.IsAvailable(node):
        return False
    if pyspin_module.IsReadable(node):
        try:
            current = str(node.GetCurrentEntry().GetSymbolic())
            if current == entry_name:
                return True
        except Exception:
            pass
    if not pyspin_module.IsWritable(node):
        return False
    try:
        entry = node.GetEntryByName(entry_name)
        if entry is None or not pyspin_module.IsReadable(entry):
            return False
        node.SetIntValue(entry.GetValue())
        return True
    except Exception:
        return False


def _execute_spinnaker_command(nodemap, pyspin_module, node_name: str) -> bool:
    node = pyspin_module.CCommandPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsWritable(node)):
        return False
    try:
        node.Execute()
        return True
    except Exception:
        return False


def _restore_spinnaker_defaults(nodemap, pyspin_module) -> bool:
    """Load factory UserSet if features got locked (common after a crashed GigE session)."""
    if not _set_spinnaker_enum(nodemap, pyspin_module, "UserSetSelector", "Default"):
        return False
    return _execute_spinnaker_command(nodemap, pyspin_module, "UserSetLoad")


def _set_spinnaker_bool(nodemap, pyspin_module, node_name: str, value: bool) -> bool:
    node = pyspin_module.CBooleanPtr(nodemap.GetNode(node_name))
    if not pyspin_module.IsAvailable(node):
        return False
    wanted = bool(value)
    if pyspin_module.IsReadable(node):
        try:
            if bool(node.GetValue()) is wanted:
                return True
        except Exception:
            pass
    if not pyspin_module.IsWritable(node):
        return False
    try:
        node.SetValue(wanted)
        return True
    except Exception:
        return False


def _set_spinnaker_int(nodemap, pyspin_module, node_name: str, value: int) -> bool:
    node = pyspin_module.CIntegerPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsWritable(node)):
        return False
    lo, hi = int(node.GetMin()), int(node.GetMax())
    inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
    inc = max(inc, 1)
    snapped = lo + ((int(value) - lo) // inc) * inc
    snapped = min(hi, max(lo, snapped))
    node.SetValue(snapped)
    return True


def _set_spinnaker_int_max(nodemap, pyspin_module, node_name: str) -> bool:
    node = pyspin_module.CIntegerPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsWritable(node)):
        return False
    node.SetValue(int(node.GetMax()))
    return True


def _read_spinnaker_string(nodemap, pyspin_module, node_name: str) -> str | None:
    node = pyspin_module.CStringPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsReadable(node)):
        return None
    try:
        return str(node.GetValue())
    except Exception:
        return None


def _read_spinnaker_enum(nodemap, pyspin_module, node_name: str) -> str | None:
    node = pyspin_module.CEnumerationPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsReadable(node)):
        return None
    try:
        return str(node.GetCurrentEntry().GetSymbolic())
    except Exception:
        return None


def _read_spinnaker_int(nodemap, pyspin_module, node_name: str) -> int | None:
    node = pyspin_module.CIntegerPtr(nodemap.GetNode(node_name))
    if not (pyspin_module.IsAvailable(node) and pyspin_module.IsReadable(node)):
        return None
    try:
        return int(node.GetValue())
    except Exception:
        return None


def _ipv4_from_int(value: int) -> str:
    v = int(value) & 0xFFFFFFFF
    return f"{(v >> 24) & 255}.{(v >> 16) & 255}.{(v >> 8) & 255}.{v & 255}"


def _apply_ccm_sensor_for_model(model: str) -> str:
    """Pick the SpinView CCM sensor preset for the connected camera."""
    text = str(model or "").upper()
    sensor = str(globals().get("CCM_SENSOR", "IMX249"))
    for key, mapped in _CCM_SENSOR_BY_MODEL.items():
        if key.upper() in text:
            sensor = mapped
            break
    globals()["CCM_SENSOR"] = sensor
    # IMX249 has no ADVANCED+GENERAL+SRGB matrix in this Spinnaker build.
    if sensor == "IMX249":
        globals()["CCM_TYPE"] = "LINEAR"
    return sensor


def _spinnaker_tl_info(cam, pyspin_module) -> dict:
    tl = cam.GetTLDeviceNodeMap()
    ip_int = _read_spinnaker_int(tl, pyspin_module, "GevDeviceIPAddress")
    dtype = _read_spinnaker_enum(tl, pyspin_module, "DeviceType") or ""
    return {
        "model": _read_spinnaker_string(tl, pyspin_module, "DeviceModelName") or "?",
        "serial": _read_spinnaker_string(tl, pyspin_module, "DeviceSerialNumber") or "?",
        "device_type": dtype,
        "ip": _ipv4_from_int(ip_int) if ip_int else "",
        "is_gige": "gige" in dtype.lower(),
    }


def _format_spinnaker_info(index: int, info: dict) -> str:
    extra = info.get("device_type") or "?"
    ip = info.get("ip") or ""
    if ip:
        extra = f"{extra} {ip}"
    return f"[{index}] {info.get('model', '?')} (SN {info.get('serial', '?')}, {extra})"


def pick_spinnaker_index(cam_list, pyspin_module, *, preferred: int = 0) -> int:
    """Choose a camera by serial, else GigE, else ``preferred`` index."""
    infos: list[tuple[int, dict]] = []
    n = cam_list.GetSize()
    for i in range(n):
        cam = cam_list[i]
        try:
            infos.append((i, _spinnaker_tl_info(cam, pyspin_module)))
        finally:
            del cam

    wanted = str(globals().get("SPINNAKER_SERIAL") or "").strip()
    if wanted:
        for i, info in infos:
            if str(info.get("serial") or "") == wanted:
                print(f"Using Spinnaker serial {wanted}: {_format_spinnaker_info(i, info)}")
                return i
        listing = ", ".join(_format_spinnaker_info(i, info) for i, info in infos) or "(none)"
        raise RuntimeError(
            f"No Spinnaker camera with serial {wanted}. Connected: {listing}"
        )

    if bool(globals().get("SPINNAKER_PREFER_GIGE", True)):
        for i, info in infos:
            if info.get("is_gige"):
                print(f"Using GigE camera: {_format_spinnaker_info(i, info)}")
                return i

    idx = int(preferred)
    if idx >= n:
        raise IndexError(f"Spinnaker camera index {idx} >= {n}")
    if infos:
        print(f"Using Spinnaker index {idx}: {_format_spinnaker_info(idx, infos[idx][1])}")
    return idx


def configure_spinnaker_gige(nodemap, stream_nodemap, pyspin_module) -> str:
    """GigE packet / throughput settings. Safe no-op on USB cameras."""
    notes: list[str] = []
    packet_node = pyspin_module.CIntegerPtr(nodemap.GetNode("GevSCPSPacketSize"))
    if not (pyspin_module.IsAvailable(packet_node) and pyspin_module.IsReadable(packet_node)):
        return ""

    requested = int(globals().get("GEV_PACKET_SIZE", 0) or 0)
    if requested > 0 and pyspin_module.IsWritable(packet_node):
        if _set_spinnaker_int(nodemap, pyspin_module, "GevSCPSPacketSize", requested):
            notes.append(f"GevSCPSPacketSize={int(packet_node.GetValue())}")
    else:
        notes.append(f"GevSCPSPacketSize={int(packet_node.GetValue())}")

    if _set_spinnaker_int_max(nodemap, pyspin_module, "DeviceLinkThroughputLimit"):
        limit = _read_spinnaker_int(nodemap, pyspin_module, "DeviceLinkThroughputLimit")
        if limit is not None:
            notes.append(f"throughput={limit}")

    _set_spinnaker_bool(stream_nodemap, pyspin_module, "StreamPacketResendEnable", True)
    _set_spinnaker_enum(stream_nodemap, pyspin_module, "StreamBufferCountMode", "Manual")
    _set_spinnaker_int(stream_nodemap, pyspin_module, "StreamBufferCountManual", 16)
    return ", ".join(notes)


def _build_ccm_settings(pyspin_module):
    settings = pyspin_module.CCMSettings()
    settings.Application = _pyspin_enum_value(
        pyspin_module, _CCM_APPLICATION[CCM_APPLICATION]
    )
    settings.Sensor = _pyspin_enum_value(pyspin_module, _CCM_SENSOR[CCM_SENSOR])
    settings.ColorTemperature = _pyspin_enum_value(
        pyspin_module, _CCM_COLOR_TEMP[CCM_COLOR_TEMP]
    )
    settings.ColorSpace = _pyspin_enum_value(
        pyspin_module, _CCM_COLOR_SPACE[CCM_COLOR_SPACE]
    )
    settings.Type = _pyspin_enum_value(pyspin_module, _CCM_TYPE[CCM_TYPE])
    return settings


def configure_spinnaker_color(nodemap, pyspin_module, *, gige: bool = False) -> str:
    """Enable auto exposure/gain/WB. GigE uses on-camera CCM; USB uses host CCM + ISP."""
    notes: list[str] = []
    if gige:
        if _set_spinnaker_enum(nodemap, pyspin_module, "ExposureAuto", "Continuous"):
            notes.append("ExpAuto")
        if _set_spinnaker_enum(nodemap, pyspin_module, "GainAuto", "Continuous"):
            notes.append("GainAuto")
        if ENABLE_AUTO_WHITE_BALANCE:
            if _set_spinnaker_enum(nodemap, pyspin_module, "BalanceWhiteAuto", "Continuous"):
                notes.append("WBAuto")
            _set_spinnaker_enum(
                nodemap, pyspin_module, "RgbTransformLightSource", "General"
            )
        color_on = bool(globals().get("ENABLE_COLOR_CORRECTION", True))
        if color_on:
            _set_spinnaker_bool(nodemap, pyspin_module, "IspEnable", True)
            pixel = _try_pixel_format(nodemap, pyspin_module)
            if pixel:
                notes.append(pixel)
            if _set_spinnaker_bool(
                nodemap, pyspin_module, "ColorTransformationEnable", True
            ):
                notes.append("ColorXF=on")
        else:
            _set_spinnaker_bool(
                nodemap, pyspin_module, "ColorTransformationEnable", False
            )
            _set_spinnaker_bool(nodemap, pyspin_module, "IspEnable", False)
        return ", ".join(notes)

    if ENABLE_AUTO_WHITE_BALANCE:
        _set_spinnaker_bool(nodemap, pyspin_module, "IspEnable", True)
        _set_spinnaker_enum(nodemap, pyspin_module, "BalanceWhiteAuto", "Continuous")
        _set_spinnaker_enum(
            nodemap, pyspin_module, "RgbTransformLightSource", "General"
        )
    _set_spinnaker_enum(nodemap, pyspin_module, "ExposureAuto", "Continuous")
    _set_spinnaker_enum(nodemap, pyspin_module, "GainAuto", "Continuous")
    if ENABLE_COLOR_CORRECTION:
        # On-host CCM replaces on-camera ColorTransformationEnable.
        _set_spinnaker_bool(nodemap, pyspin_module, "ColorTransformationEnable", False)
    return "USB ISP/CCM"


def _try_pixel_format(nodemap, pyspin_module, names: tuple[str, ...] = ("RGB8", "RGB8Packed", "BGR8")) -> str | None:
    for name in names:
        if _set_spinnaker_enum(nodemap, pyspin_module, "PixelFormat", name):
            return name
    return None


def _read_spinnaker_pixel_format(nodemap, pyspin_module) -> str:
    """Read PixelFormat only — do not change it unless color setup already did."""
    node = pyspin_module.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if pyspin_module.IsAvailable(node) and pyspin_module.IsReadable(node):
        try:
            return str(node.GetCurrentEntry().GetSymbolic())
        except Exception:
            pass
    return "unknown"


def configure_spinnaker_capture_size(nodemap, pyspin_module) -> str:
    """Use the full sensor (optional binning only). Do not center-crop Width."""
    notes: list[str] = []
    binning = int(globals().get("CAMERA_BINNING", 1))
    if binning > 1:
        binned = _set_spinnaker_int(nodemap, pyspin_module, "BinningHorizontal", binning)
        binned = _set_spinnaker_int(nodemap, pyspin_module, "BinningVertical", binning) and binned
        if binned:
            notes.append(f"binning {binning}x{binning}")
        else:
            notes.append("full-res grab")
    else:
        _set_spinnaker_int(nodemap, pyspin_module, "BinningHorizontal", 1)
        _set_spinnaker_int(nodemap, pyspin_module, "BinningVertical", 1)
        notes.append("full-res grab")

    for name in ("OffsetX", "OffsetY"):
        _set_spinnaker_int(nodemap, pyspin_module, name, 0)

    w_node = pyspin_module.CIntegerPtr(nodemap.GetNode("Width"))
    h_node = pyspin_module.CIntegerPtr(nodemap.GetNode("Height"))
    if pyspin_module.IsAvailable(w_node) and pyspin_module.IsWritable(w_node):
        w_node.SetValue(int(w_node.GetMax()))
        notes.append(f"Width={int(w_node.GetValue())}")
    if pyspin_module.IsAvailable(h_node) and pyspin_module.IsWritable(h_node):
        h_node.SetValue(int(h_node.GetMax()))
        notes.append(f"Height={int(h_node.GetValue())}")
    return ", ".join(notes) or "default size"


def _downscale_rgb(rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    max_width = max(1, int(max_width))
    if w <= max_width:
        return rgb
    nh = max(1, int(round(h * (max_width / float(w)))))
    return cv2.resize(rgb, (max_width, nh), interpolation=cv2.INTER_AREA)


def list_dshow_device_names(max_index: int = 12) -> list[str]:
    """Best-effort DirectShow capture names via PowerShell (Windows)."""
    if sys.platform != "win32":
        return []
    import subprocess

    script = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -eq 'Camera' -and $_.Status -eq 'OK' } | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return []
    names = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return names[:max_index]


def find_opencv_camera_index(
    preferred_name: str | None = None,
    *,
    fallback: int = 0,
    max_index: int = 12,
) -> tuple[int, str | None]:
    """Resolve a DirectShow index. Match ``preferred_name`` when possible."""
    wanted = (preferred_name or "").strip()
    names = list_dshow_device_names(max_index=max_index)
    if wanted and names:
        lower = wanted.lower()
        for i, name in enumerate(names):
            if lower in name.lower():
                return i, name
    if wanted:
        # Some OpenCV builds accept "video=<FriendlyName>"; probe once.
        probe = cv2.VideoCapture(f"video={wanted}", cv2.CAP_DSHOW)
        ok = bool(probe.isOpened())
        probe.release()
        if ok:
            return fallback, wanted
    if names and 0 <= fallback < len(names):
        return fallback, names[fallback]
    return fallback, None


def _fourcc_mjpeg() -> int:
    return cv2.VideoWriter_fourcc(*"MJPG")


def configure_opencv_capture(cap: cv2.VideoCapture, *, width: int, height: int) -> str:
    """Request MJPEG + low latency for the ELP UVC global shutter."""
    notes: list[str] = []
    try:
        if cap.set(cv2.CAP_PROP_FOURCC, _fourcc_mjpeg()):
            notes.append("MJPEG")
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if width > 0 and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    notes.append(f"{got_w}x{got_h}")
    return ", ".join(notes)


def _opencv_capture_size(*, high: bool) -> tuple[int, int]:
    size = (
        tuple(globals().get("OPENCV_HQ_SIZE", (1920, 1200)))
        if high
        else tuple(globals().get("OPENCV_FAST_SIZE", (1280, 720)))
    )
    return int(size[0]), int(size[1])


class LiveCamera:
    """Live camera via Spinnaker, OpenCV DirectShow, or a video file."""

    def __init__(
        self,
        backend: str,
        *,
        index: int = 0,
        spinnaker_index: int = 0,
        video_path: str | Path | None = None,
    ):
        self.backend = backend
        self.index = index
        self.spinnaker_index = spinnaker_index
        self.video_path = Path(video_path) if video_path else None
        self._cap = None
        self._system = None
        self._cam_list = None
        self._cam = None
        self._processor = None
        self._ccm_settings = None
        self.is_gige = False
        self._enum_cache: dict[str, str] = {}
        self.last_native_hw: tuple[int, int] | None = None
        self.last_grab_ms = 0.0
        self.opencv_device_name: str | None = None

    def open(self) -> None:
        if self.backend == "video":
            path = self.video_path or Path(globals().get("VIDEO_PATH", ""))
            path = Path(path)
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            print(f"Video source: {path}")
            return

        if self.backend == "opencv":
            preferred = str(globals().get("OPENCV_CAMERA_NAME", "") or "").strip()
            index, name = find_opencv_camera_index(
                preferred, fallback=int(self.index)
            )
            self.index = index
            self.opencv_device_name = name or preferred or None
            # OpenCV 5 on this machine cannot open DirectShow by friendly name;
            # use the matched numeric index (fallback CAMERA_INDEX).
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not self._cap.isOpened() and preferred:
                self._cap.release()
                self._cap = cv2.VideoCapture(f"video={preferred}", cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"OpenCV could not open camera index {self.index}"
                    + (f" ({self.opencv_device_name})" if self.opencv_device_name else "")
                )
            high = not bool(globals().get("FAST_LIVE", False))
            width, height = _opencv_capture_size(high=high)
            note = configure_opencv_capture(self._cap, width=width, height=height)
            label = self.opencv_device_name or f"index {self.index}"
            print(f"OpenCV camera: {label} @ {note}")
            return

        if self.backend == "spinnaker":
            import PySpin

            self._system = PySpin.System.GetInstance()
            self._cam_list = self._system.GetCameras()
            n = self._cam_list.GetSize()
            if n == 0:
                self._release_spinnaker()
                raise RuntimeError("No Spinnaker cameras detected.")
            try:
                self.spinnaker_index = pick_spinnaker_index(
                    self._cam_list, PySpin, preferred=self.spinnaker_index
                )
            except Exception:
                self._release_spinnaker()
                raise

            self._cam = self._cam_list[self.spinnaker_index]
            self._cam.Init()
            nodemap = self._cam.GetNodeMap()
            tl_info = _spinnaker_tl_info(self._cam, PySpin)
            self.is_gige = bool(tl_info.get("is_gige"))
            self._enum_cache = {}
            ccm_sensor = _apply_ccm_sensor_for_model(str(tl_info.get("model") or ""))

            stream_nodemap = self._cam.GetTLStreamNodeMap()
            node_mode = PySpin.CEnumerationPtr(stream_nodemap.GetNode("StreamBufferHandlingMode"))
            if PySpin.IsReadable(node_mode) and PySpin.IsWritable(node_mode):
                newest = node_mode.GetEntryByName("NewestOnly")
                if PySpin.IsReadable(newest):
                    node_mode.SetIntValue(newest.GetValue())

            acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
            if PySpin.IsReadable(acq_mode) and PySpin.IsWritable(acq_mode):
                continuous = acq_mode.GetEntryByName("Continuous")
                if PySpin.IsReadable(continuous):
                    acq_mode.SetIntValue(continuous.GetValue())

            if self.is_gige:
                isp_node = PySpin.CBooleanPtr(nodemap.GetNode("IspEnable"))
                if (
                    PySpin.IsAvailable(isp_node)
                    and PySpin.IsReadable(isp_node)
                    and bool(isp_node.GetValue())
                    and not PySpin.IsWritable(isp_node)
                ):
                    if _restore_spinnaker_defaults(nodemap, PySpin):
                        print("GigE camera was locked; loaded factory UserSet Default")

            gige_note = configure_spinnaker_gige(nodemap, stream_nodemap, PySpin)
            color_note = configure_spinnaker_color(
                nodemap, PySpin, gige=self.is_gige
            )
            size_note = configure_spinnaker_capture_size(nodemap, PySpin)
            pixel_format = _read_spinnaker_pixel_format(nodemap, PySpin)
            extras = [size_note]
            if gige_note:
                extras.append(gige_note)
            if color_note:
                extras.append(color_note)
            extras.append(f"CCM={ccm_sensor}")
            print(f"Spinnaker capture: PixelFormat={pixel_format}, {', '.join(extras)}")

            self._processor = PySpin.ImageProcessor()
            self.set_quality_mode(True)
            if not self.is_gige:
                self.set_color_correction(bool(globals().get("ENABLE_COLOR_CORRECTION", True)))
            for name in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto"):
                self.get_enum(name)
            self._cam.BeginAcquisition()
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def set_quality_mode(self, high: bool) -> None:
        """Switch grab quality. OpenCV retargets MJPEG size; Spinnaker keeps HQ demosaic."""
        if self.backend == "opencv" and self._cap is not None:
            width, height = _opencv_capture_size(high=bool(high))
            note = configure_opencv_capture(self._cap, width=width, height=height)
            print(f"OpenCV quality: {'HQ' if high else 'FAST'} ({note})")
            return
        if self.backend != "spinnaker" or self._processor is None:
            return
        import PySpin

        try:
            self._processor.SetColorProcessing(
                PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR
            )
        except Exception:
            pass

    def set_color_correction(self, enabled: bool) -> None:
        """On-camera ColorTransformation for GigE; host SpinView CCM for USB."""
        globals()["ENABLE_COLOR_CORRECTION"] = bool(enabled)
        if self.backend != "spinnaker":
            return
        import PySpin

        if self.is_gige:
            self._ccm_settings = None
            if self._cam is None:
                return
            nodemap = self._cam.GetNodeMap()
            if _set_spinnaker_bool(nodemap, PySpin, "ColorTransformationEnable", bool(enabled)):
                return
            streaming = False
            try:
                streaming = bool(self._cam.IsStreaming())
            except Exception:
                streaming = True
            if streaming:
                try:
                    self._cam.EndAcquisition()
                except Exception:
                    pass
            if enabled:
                _set_spinnaker_bool(nodemap, PySpin, "IspEnable", True)
                _try_pixel_format(nodemap, PySpin)
            _set_spinnaker_bool(nodemap, PySpin, "ColorTransformationEnable", bool(enabled))
            if streaming:
                try:
                    self._cam.BeginAcquisition()
                except Exception as exc:
                    print(f"Could not restart GigE stream after color toggle: {exc}")
            return

        if enabled:
            if str(globals().get("CCM_SENSOR", "")).upper() == "IMX249":
                self._ccm_settings = None
                return
            if self._ccm_settings is None:
                try:
                    self._ccm_settings = _build_ccm_settings(PySpin)
                except Exception:
                    self._ccm_settings = None
        else:
            self._ccm_settings = None

    def read_rgb(self) -> np.ndarray | None:
        t0 = time.perf_counter()
        max_width = int(globals().get("FRAME_MAX_WIDTH", 1280))
        if self.backend in {"opencv", "video"}:
            ok, frame_bgr = self._cap.read()
            if not ok and self.backend == "video" and self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = self._cap.read()
            if not ok:
                return None
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.last_native_hw = (int(rgb.shape[1]), int(rgb.shape[0]))
            out = _downscale_rgb(rgb, max_width)
            self.last_grab_ms = (time.perf_counter() - t0) * 1000.0
            return out

        if self.backend == "spinnaker":
            import PySpin

            image_result = self._cam.GetNextImage(1000)
            try:
                if image_result.IsIncomplete():
                    return None
                converted = self._processor.Convert(image_result, PySpin.PixelFormat_RGB8)
                if self._ccm_settings is not None:
                    try:
                        converted = PySpin.ImageUtilityCCM_CreateColorCorrected(
                            converted, self._ccm_settings
                        )
                    except Exception as exc:
                        print(f"Color correction unsupported on this camera ({exc}); demosaic only.")
                        self._ccm_settings = None
                rgb = np.asarray(converted.GetNDArray(), dtype=np.uint8).copy()
                self.last_native_hw = (int(rgb.shape[1]), int(rgb.shape[0]))
                out = _downscale_rgb(rgb, max_width)
                self.last_grab_ms = (time.perf_counter() - t0) * 1000.0
                return out
            finally:
                image_result.Release()

        return None

    def get_enum(self, name: str) -> str | None:
        if self.backend != "spinnaker" or self._cam is None:
            return None
        import PySpin

        node = PySpin.CEnumerationPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
            return None
        try:
            value = str(node.GetCurrentEntry().GetSymbolic())
            self._enum_cache[name] = value
            return value
        except Exception:
            return None

    def set_enum(self, name: str, entry: str) -> bool:
        if self.backend != "spinnaker" or self._cam is None:
            return False
        if self._enum_cache.get(name) == entry:
            return True
        import PySpin

        ok = _set_spinnaker_enum(self._cam.GetNodeMap(), PySpin, name, entry)
        if ok:
            self._enum_cache[name] = entry
        return ok

    def get_float(self, name: str) -> float | None:
        if self.backend != "spinnaker" or self._cam is None:
            return None
        import PySpin

        node = PySpin.CFloatPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
            return None
        try:
            return float(node.GetValue())
        except Exception:
            return None

    def set_float(self, name: str, value: float) -> bool:
        if self.backend != "spinnaker" or self._cam is None:
            return False
        import PySpin

        node = PySpin.CFloatPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
            return False
        try:
            lo, hi = float(node.GetMin()), float(node.GetMax())
            node.SetValue(min(hi, max(lo, float(value))))
            return True
        except Exception:
            return False

    def float_range(self, name: str) -> tuple[float, float] | None:
        if self.backend != "spinnaker" or self._cam is None:
            return None
        import PySpin

        node = PySpin.CFloatPtr(self._cam.GetNodeMap().GetNode(name))
        if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
            return None
        try:
            return float(node.GetMin()), float(node.GetMax())
        except Exception:
            return None

    def get_balance_ratio(self, selector: str) -> float | None:
        """Read one WB channel. Avoid this while BalanceWhiteAuto is Continuous."""
        if not self.set_enum("BalanceRatioSelector", selector):
            return None
        return self.get_float("BalanceRatio")

    def set_balance_ratio(self, selector: str, value: float) -> bool:
        if not self.set_enum("BalanceRatioSelector", selector):
            return False
        return self.set_float("BalanceRatio", value)

    def is_auto_on(self, enum_name: str) -> bool:
        current = self.get_enum(enum_name)
        return current is not None and current.lower() == "continuous"

    def _release_spinnaker(self) -> None:
        if self._cam is not None:
            try:
                self._cam.EndAcquisition()
            except Exception:
                pass
            try:
                self._cam.DeInit()
            except Exception:
                pass
            del self._cam
            self._cam = None
        if self._cam_list is not None:
            self._cam_list.Clear()
            del self._cam_list
            self._cam_list = None
        if self._system is not None:
            self._system.ReleaseInstance()
            del self._system
            self._system = None
        self._processor = None
        self._ccm_settings = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.backend == "spinnaker":
            self._release_spinnaker()


def list_spinnaker_cameras() -> list[str]:
    """List cameras without Init/ReleaseInstance (safe to call before opening)."""
    try:
        import PySpin
    except ImportError:
        return []

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    names: list[str] = []
    try:
        for i in range(cam_list.GetSize()):
            cam = cam_list[i]
            try:
                names.append(_format_spinnaker_info(i, _spinnaker_tl_info(cam, PySpin)))
            except Exception as exc:
                names.append(f"[{i}] <error: {exc}>")
            finally:
                del cam
    finally:
        cam_list.Clear()
        del cam_list
    return names


def list_opencv_cameras(max_index: int = 8) -> list[str]:
    found: list[str] = []
    names = list_dshow_device_names(max_index=max_index)
    preferred = str(globals().get("OPENCV_CAMERA_NAME", "") or "").strip()
    if preferred:
        found.append(f"[name] {preferred}")
    for i, name in enumerate(names):
        if preferred and preferred.lower() in name.lower():
            continue
        found.append(f"[{i}] {name}")
    if found:
        return found

    prev_log = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    try:
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append(f"[{i}] {w}x{h}")
            cap.release()
    finally:
        if prev_log is not None:
            cv2.setLogLevel(prev_log)
    return found


def print_cameras(*, include_opencv: bool = False, include_spinnaker: bool = True) -> None:
    if include_spinnaker:
        print("Spinnaker cameras:", list_spinnaker_cameras() or ["(none)"])
    if include_opencv:
        print("OpenCV cameras:", list_opencv_cameras() or ["(none)"])


def prepare() -> None:
    """Print env, verify PySpin if needed, and list cameras."""
    print_env()
    backend = str(globals().get("CAMERA_BACKEND", "opencv")).lower()
    if backend == "spinnaker":
        require_pyspin()
        # Do not enumerate DirectShow devices here: NVIDIA's virtual camera
        # driver dumps a wall of VCAMDS INFO/WARN/ERR lines into the console.
        print_cameras(include_opencv=False, include_spinnaker=True)
    elif backend == "video":
        path = Path(globals().get("VIDEO_PATH", ""))
        print("Skipping PySpin check (video backend).")
        print(f"Video source: {path}  exists={path.is_file()}")
    else:
        print("Skipping PySpin check (opencv backend).")
        print_cameras(include_opencv=True, include_spinnaker=False)


def _ensure_cascade_importable() -> None:
    """Prefer installed napari-pecan-py; fall back to sibling plugin checkout."""
    try:
        import pecanpy.widgets.cascade_seg.model  # noqa: F401
        return
    except ImportError:
        pass
    candidates = [
        REPO_ROOT / "src",
        REPO_ROOT.parent / "napari-pecan-py" / "src",
        REPO_ROOT.parent / "pecan_py_napari" / "src",
        Path(os.environ["PECAN_PLUGIN_SRC"])
        if os.environ.get("PECAN_PLUGIN_SRC")
        else None,
    ]
    for src in candidates:
        if src is None:
            continue
        src_s = str(src)
        if src.is_dir() and src_s not in sys.path:
            sys.path.insert(0, src_s)
            try:
                import pecanpy.widgets.cascade_seg.model  # noqa: F401
                return
            except ImportError:
                continue
    raise ImportError(
        "Cascade backend needs napari-pecan-py.\n"
        "  pip install -e /path/to/napari-pecan-py\n"
        "  or set PECAN_PLUGIN_SRC to that repo's src/ directory.\n"
        "Pure YOLO mode does not need the plugin."
    )


def _print_device() -> None:
    print(f"Classes: {class_names}")
    print(f"Device: {resolved_device}")
    try:
        import torch

        if str(resolved_device).lower() not in {"cpu"} and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass


def load_yolo_model():
    """Load YOLO-seg weights, fuse, and cache ``model`` / ``class_names`` / ``resolved_device``."""
    global model, resolved_device, class_names
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found:\n  {WEIGHTS_PATH}\n\n"
            "Put a .pt under weights/ (e.g. 2026-08-19_131846.pt or "
            "halfyield-model.pt), or set PECAN_YOLO_WEIGHTS."
        )

    resolved_device = resolve_yolo_device(DEVICE)
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS_PATH))
    try:
        model.fuse()
    except Exception:
        pass
    class_names = {int(k): str(v) for k, v in model.names.items()}
    _print_device()
    return model, class_names, resolved_device


def load_cascade_model():
    """Load cascaded U-Net weights used by the Segmentation widget."""
    global model, resolved_device, class_names
    global cascade_image_size, cascade_torch_device, label_id_by_class
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Cascade weights not found:\n  {WEIGHTS_PATH}")

    _ensure_cascade_importable()
    from pecanpy.widgets.cascade_seg.model import (
        _predict_frame_label_map,
        _resolve_torch_device,
        load_cascade_checkpoint,
    )

    model, ckpt = load_cascade_checkpoint(WEIGHTS_PATH)
    cascade_torch_device = _resolve_torch_device(DEVICE)
    model.to(cascade_torch_device)
    model.eval()
    cascade_image_size = int(ckpt.get("image_size", 384))
    label_id_by_class = {
        str(k): int(v) for k, v in dict(ckpt.get("label_id_by_class") or {}).items()
    }
    class_names = {
        int(v): str(k)
        for k, v in label_id_by_class.items()
        if str(k).strip().lower() != "damaged kernel"
    }
    resolved_device = str(cascade_torch_device)
    print(
        f"Cascade: {ckpt.get('architecture')} / {ckpt.get('encoder_name')} "
        f"imgsz={cascade_image_size} stages={ckpt.get('trained_stages')}"
    )
    _print_device()
    dummy = np.zeros((cascade_image_size, cascade_image_size, 3), dtype=np.uint8)
    _predict_frame_label_map(
        model, dummy, image_size=cascade_image_size, device=cascade_torch_device
    )
    return model, class_names, resolved_device


def load_model():
    """Load YOLO or cascade weights depending on ``MODEL_BACKEND``."""
    backend = str(globals().get("MODEL_BACKEND", "yolo")).lower()
    if backend == "cascade":
        return load_cascade_model()
    return load_yolo_model()


def predict_cascade_label_map(frame_rgb: np.ndarray) -> np.ndarray:
    from pecanpy.widgets.cascade_seg.model import _predict_frame_label_map

    return _predict_frame_label_map(
        model,
        frame_rgb,
        image_size=int(cascade_image_size),
        device=cascade_torch_device,
    )


def _class_key(name: str) -> str:
    return str(name).strip().lower()


def _draw_priority(name: str) -> int:
    """Pecan first, kernel on top, everything else in between."""
    key = _class_key(name)
    if key == "pecan":
        return 0
    if key == "kernel":
        return 2
    return 1


def _normalize_roi(roi, w: int, h: int):
    if roi is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def overlay_masks(
    frame_bgr: np.ndarray,
    result,
    names: dict[int, str],
    colors_bgr: dict[str, tuple[int, int, int]],
    alpha: float,
    roi=None,
) -> np.ndarray:
    """Tint instance masks at ``alpha``. If ``roi`` is set, only paint inside it."""
    out = frame_bgr.copy()
    if result is None or result.masks is None or result.boxes is None:
        return out

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    h, w = out.shape[:2]
    roi = _normalize_roi(roi, w, h)
    overlay = out.copy()

    order = sorted(
        range(len(masks)),
        key=lambda i: _draw_priority(names.get(int(classes[i]), "")),
    )
    for i in order:
        name = names.get(int(classes[i]), str(int(classes[i])))
        color = colors_bgr.get(_class_key(name), (0, 255, 255))
        mask = masks[i]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        keep = mask > 0.5
        if roi is not None:
            x1, y1, x2, y2 = roi
            inside = np.zeros_like(keep)
            inside[y1:y2, x1:x2] = keep[y1:y2, x1:x2]
            if not np.any(inside):
                continue
            keep = inside
        overlay[keep] = color

    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, dst=out)
    return out


def overlay_label_map(
    frame_bgr: np.ndarray,
    label_map: np.ndarray | None,
    names: dict[int, str],
    colors_bgr: dict[str, tuple[int, int, int]],
    alpha: float,
    roi=None,
) -> np.ndarray:
    """Tint semantic label-map pixels (cascade / U-Net)."""
    out = frame_bgr.copy()
    if label_map is None:
        return out
    h, w = out.shape[:2]
    lm = np.asarray(label_map)
    if lm.shape[0] != h or lm.shape[1] != w:
        lm = cv2.resize(lm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    roi = _normalize_roi(roi, w, h)
    overlay = out.copy()
    order = sorted(names.items(), key=lambda kv: _draw_priority(kv[1]))
    for label_id, name in order:
        color = colors_bgr.get(_class_key(name), (0, 255, 255))
        keep = lm == int(label_id)
        if roi is not None:
            x1, y1, x2, y2 = roi
            inside = np.zeros_like(keep)
            inside[y1:y2, x1:x2] = keep[y1:y2, x1:x2]
            keep = inside
        if not np.any(keep):
            continue
        overlay[keep] = color
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, dst=out)
    return out


def _class_union_masks(result, names: dict[int, str], h: int, w: int, roi=None):
    """Boolean pecan / kernel unions from YOLO instance masks, resized to the frame."""
    pecan = np.zeros((h, w), dtype=bool)
    kernel = np.zeros((h, w), dtype=bool)
    if result is None or result.masks is None or result.boxes is None:
        return pecan, kernel
    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    roi = _normalize_roi(roi, w, h)
    for i, cls in enumerate(classes):
        key = _class_key(names.get(int(cls), ""))
        mask = masks[i]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        keep = mask > 0.5
        if roi is not None:
            x1, y1, x2, y2 = roi
            clipped = np.zeros_like(keep)
            clipped[y1:y2, x1:x2] = keep[y1:y2, x1:x2]
            keep = clipped
        if key == "pecan":
            pecan |= keep
        elif key == "kernel":
            kernel |= keep
    return pecan, kernel


def _stats_from_pecan_kernel(pecan: np.ndarray, kernel: np.ndarray) -> dict:
    min_area = int(globals().get("MIN_PECAN_AREA", 80))
    overlap_frac = float(globals().get("CRACK_OVERLAP_MIN", 0.02))
    n_pecan = n_cracked = n_uncracked = 0
    # Shell and kernel masks are usually adjacent, not overlapping.
    kernel_touch = cv2.dilate(
        kernel.astype(np.uint8),
        np.ones((15, 15), np.uint8),
        iterations=1,
    ).astype(bool)
    n_labels, labels = cv2.connectedComponents(pecan.astype(np.uint8))
    for lab in range(1, n_labels):
        inst = labels == lab
        area = int(inst.sum())
        if area < min_area:
            continue
        n_pecan += 1
        overlap = int((inst & kernel_touch).sum())
        if overlap >= max(1, int(overlap_frac * area)):
            n_cracked += 1
        else:
            n_uncracked += 1
    label_px = int((pecan | kernel).sum())
    kernel_px = int(kernel.sum())
    ratio = (kernel_px / label_px) if label_px else 0.0
    return {
        "n_pecan": n_pecan,
        "n_uncracked": n_uncracked,
        "n_cracked": n_cracked,
        "kernel_label_ratio": ratio,
        "kernel_px": kernel_px,
        "label_px": label_px,
    }


def compute_shell_stats(result, names: dict[int, str], h: int, w: int, roi=None) -> dict:
    """Count pecan blobs and crack exposure from Kernel vs Pecan masks.

    Pecans are connected components of the pecan union. A blob is cracked if
    enough of it overlaps kernel. Ratio is kernel area / all labeled area.
    """
    pecan, kernel = _class_union_masks(result, names, h, w, roi=roi)
    return _stats_from_pecan_kernel(pecan, kernel)


def compute_shell_stats_from_label_map(label_map, h: int, w: int, roi=None) -> dict:
    ids = globals().get("label_id_by_class") or {"Pecan": 3, "Kernel": 2}
    pecan_id = int(ids.get("Pecan", 3))
    kernel_id = int(ids.get("Kernel", 2))
    pecan = np.zeros((h, w), dtype=bool)
    kernel = np.zeros((h, w), dtype=bool)
    if label_map is not None:
        lm = np.asarray(label_map)
        if lm.shape[0] != h or lm.shape[1] != w:
            lm = cv2.resize(lm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        roi = _normalize_roi(roi, w, h)
        pecan = lm == pecan_id
        kernel = lm == kernel_id
        if roi is not None:
            x1, y1, x2, y2 = roi
            clipped_p = np.zeros_like(pecan)
            clipped_k = np.zeros_like(kernel)
            clipped_p[y1:y2, x1:x2] = pecan[y1:y2, x1:x2]
            clipped_k[y1:y2, x1:x2] = kernel[y1:y2, x1:x2]
            pecan, kernel = clipped_p, clipped_k
    return _stats_from_pecan_kernel(pecan, kernel)


def draw_stats_hud(frame_bgr: np.ndarray, stats: dict) -> np.ndarray:
    """Pecan / crack counts in the bottom-left corner."""
    h, _w = frame_bgr.shape[:2]
    ratio = float(stats.get("kernel_label_ratio") or 0.0)
    lines = [
        f"Pecans: {int(stats.get('n_pecan') or 0)}",
        f"Uncracked: {int(stats.get('n_uncracked') or 0)}",
        f"Cracked: {int(stats.get('n_cracked') or 0)}",
        f"Kernel/labels: {100.0 * ratio:.1f}%",
    ]
    y = h - 12 - 22 * (len(lines) - 1)
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return frame_bgr


def draw_legend(frame_bgr: np.ndarray, names: dict[int, str]) -> np.ndarray:
    """Paint a small class-color key plus ROI hint."""
    y = 28
    seen: set[str] = set()
    for name in names.values():
        key = _class_key(name)
        if key in seen:
            continue
        seen.add(key)
        color = CLASS_COLORS_BGR.get(key, (0, 255, 255))
        cv2.rectangle(frame_bgr, (12, y - 16), (32, y + 2), color, thickness=-1)
        cv2.putText(
            frame_bgr,
            name,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 26
    cv2.putText(
        frame_bgr,
        "drag ROI  |  c clear  |  q quit",
        (12, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame_bgr


def draw_fps_hud(
    frame_bgr: np.ndarray,
    fps: float,
    fps_max: float,
    device_text: str,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    """Draw measured pipeline FPS and a timing breakdown in the top-right corner."""
    _h, w = frame_bgr.shape[:2]
    lines = [f"{fps:.1f} fps", f"max {fps_max:.1f}", device_text]
    if extra_lines:
        lines.extend(extra_lines)
    y = 28
    for i, line in enumerate(lines):
        scale = 0.7 if i < 3 else 0.5
        thick = 2 if i < 3 else 1
        (tw, _th), _base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = max(12, w - tw - 14)
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thick + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            thick,
            cv2.LINE_AA,
        )
        y += 26 if i < 3 else 20
    return frame_bgr


def _mouse_to_image(x: int, y: int, view: dict) -> tuple[int, int]:
    """Map mouse coords on the shown (scaled) frame back to camera pixels."""
    scale = float(view.get("scale") or 1.0)
    if scale <= 0:
        scale = 1.0
    ox = float(view.get("ox") or 0.0)
    oy = float(view.get("oy") or 0.0)
    ix = int(round((x - ox) / scale))
    iy = int(round((y - oy) / scale))
    w = max(1, int(view.get("w") or 1))
    h = max(1, int(view.get("h") or 1))
    return int(np.clip(ix, 0, w - 1)), int(np.clip(iy, 0, h - 1))


def _letterbox_to_size(frame_bgr: np.ndarray, out_w: int, out_h: int):
    """Fit ``frame_bgr`` inside ``out_w x out_h`` with black bars; keep aspect ratio."""
    h, w = frame_bgr.shape[:2]
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    scale = min(out_w / float(w), out_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = (
        frame_bgr
        if nw == w and nh == h
        else cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    )
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ox = (out_w - nw) // 2
    oy = (out_h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, float(scale), float(ox), float(oy)


def _video_window_size(win_name: str, view: dict, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Window client size for letterboxing, ignoring title-bar jitter."""
    default_w = min(int(globals().get("DISPLAY_MAX_WIDTH", 960)), max(1, int(frame_w)))
    default_h = max(1, int(round(default_w * frame_h / float(frame_w))))
    fallback_w = int(view.get("disp_w") or default_w)
    fallback_h = int(view.get("disp_h") or default_h)
    try:
        _x, _y, ww, wh = cv2.getWindowImageRect(win_name)
        ww, wh = int(ww), int(wh)
    except Exception:
        return fallback_w, fallback_h
    if ww < 64 or wh < 64:
        return fallback_w, fallback_h
    last_w = int(view.get("disp_w") or ww)
    last_h = int(view.get("disp_h") or wh)
    if abs(ww - last_w) <= 16 and 8 <= (wh - last_h) <= 56:
        return last_w, last_h
    if abs(ww - last_w) < 8 and abs(wh - last_h) < 8:
        return last_w, last_h
    return ww, wh


def _log_to_pos(value: float, lo: float, hi: float, steps: int) -> int:
    value = min(hi, max(lo, float(value)))
    t = (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return int(np.clip(round(t * steps), 0, steps))


def _pos_to_log(pos: int, lo: float, hi: float, steps: int) -> float:
    t = min(steps, max(0, int(pos))) / float(steps)
    return float(np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo))))


def apply_quality_preset(high: bool) -> None:
    """Update live working size / YOLO size. Sensor binning stays as opened."""
    g = globals()
    g["FAST_LIVE"] = not high
    if high:
        g["FRAME_MAX_WIDTH"] = int(g.get("HQ_FRAME_MAX_WIDTH", 1920))
        g["INFER_IMGSZ"] = int(g.get("HQ_INFER_IMGSZ", 1280))
    else:
        g["FRAME_MAX_WIDTH"] = int(g.get("FAST_FRAME_MAX_WIDTH", 960))
        g["INFER_IMGSZ"] = int(g.get("FAST_INFER_IMGSZ", 640))


class CameraControlPanel:
    """Trackbars for analog camera settings; checkboxes for on/off switches."""

    WINDOW_WIDTH = 520
    WINDOW_HEIGHT = 420
    PANEL_W = 440
    PANEL_H = 360

    EXP_STEPS = 1000
    GAIN_STEPS = 270
    WB_STEPS = 400
    BOX = 18
    CHECKS = (
        ("hq_grab", "HQ Grab"),
        ("color_corr", "Color Corr"),
        ("exp_auto", "Exp Auto"),
        ("gain_auto", "Gain Auto"),
        ("wb_auto", "WB Auto"),
    )

    def __init__(self, camera: LiveCamera, default_fps: int):
        self.camera = camera
        self.win = CONTROLS_WINDOW_NAME
        self._last_auto_refresh = 0.0
        self._exp_range = (100.0, 20000.0)
        self._gain_range = (0.0, 27.0)
        self._wb_range = (0.25, 4.0)
        self._hitboxes: list[tuple[str, int, int, int, int]] = []
        self._last_manual: dict[str, int] = {}
        self._prev_autos: dict[str, bool] = {}
        self._check_clicked = False
        self._cached_exp: float | None = None
        self._cached_gain: float | None = None
        self._cached_red: float | None = None
        self._cached_blue: float | None = None
        self.checks = {
            "hq_grab": not bool(globals().get("FAST_LIVE", False)),
            "color_corr": bool(globals().get("ENABLE_COLOR_CORRECTION", True)),
            "exp_auto": True,
            "gain_auto": True,
            "wb_auto": bool(globals().get("ENABLE_AUTO_WHITE_BALANCE", True)),
        }
        self._quality_high = bool(self.checks["hq_grab"])
        self._color_corr = bool(self.checks["color_corr"])
        probed = camera.float_range("ExposureTime")
        if probed is not None:
            lo, hi = probed
            self._exp_range = (max(lo, 20.0), min(hi, 100000.0))
        probed = camera.float_range("Gain")
        if probed is not None:
            self._gain_range = probed
        probed = camera.float_range("BalanceRatio")
        if probed is not None:
            self._wb_range = probed

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        dummy = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        cv2.imshow(self.win, dummy)

        fps0 = int(np.clip(default_fps, 1, 30))
        cv2.createTrackbar("FPS", self.win, fps0, 30, lambda _v: None)
        cv2.createTrackbar("Exposure", self.win, 0, self.EXP_STEPS, lambda _v: None)
        cv2.createTrackbar("Gain", self.win, 0, self.GAIN_STEPS, lambda _v: None)
        cv2.createTrackbar("WB Red", self.win, 0, self.WB_STEPS, lambda _v: None)
        cv2.createTrackbar("WB Blue", self.win, 0, self.WB_STEPS, lambda _v: None)
        cv2.resizeWindow(self.win, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        cv2.setMouseCallback(self.win, self._on_mouse)
        self._pull_autos_from_camera()
        self._prev_autos = {
            "exp_auto": bool(self.checks["exp_auto"]),
            "gain_auto": bool(self.checks["gain_auto"]),
            "wb_auto": bool(self.checks["wb_auto"]),
        }
        self._pull_values_from_camera()
        self.render()

    def target_fps(self) -> int:
        return max(1, cv2.getTrackbarPos("FPS", self.win))

    def _lin_to_pos(self, value: float, lo: float, hi: float, steps: int) -> int:
        if hi <= lo:
            return 0
        t = (float(value) - lo) / (hi - lo)
        return int(np.clip(round(t * steps), 0, steps))

    def _pos_to_lin(self, pos: int, lo: float, hi: float, steps: int) -> float:
        t = min(steps, max(0, int(pos))) / float(steps)
        return float(lo + t * (hi - lo))

    def _safe_set_pos(self, name: str, pos: int) -> None:
        try:
            cv2.setTrackbarPos(name, self.win, int(pos))
        except Exception:
            pass

    def _pull_autos_from_camera(self) -> None:
        if self.camera.backend != "spinnaker":
            return
        mapping = (
            ("exp_auto", "ExposureAuto"),
            ("gain_auto", "GainAuto"),
            ("wb_auto", "BalanceWhiteAuto"),
        )
        for key, node in mapping:
            current = self.camera.get_enum(node)
            if current is not None:
                self.checks[key] = current.lower() == "continuous"

    def _pull_values_from_camera(self) -> None:
        if self.camera.backend != "spinnaker":
            return
        exp = self.camera.get_float("ExposureTime")
        if exp is not None:
            self._cached_exp = exp
            if self.checks["exp_auto"]:
                self._safe_set_pos(
                    "Exposure",
                    _log_to_pos(exp, *self._exp_range, self.EXP_STEPS),
                )
        gain = self.camera.get_float("Gain")
        if gain is not None:
            self._cached_gain = gain
            if self.checks["gain_auto"]:
                self._safe_set_pos(
                    "Gain",
                    self._lin_to_pos(gain, *self._gain_range, self.GAIN_STEPS),
                )
        # Switching BalanceRatioSelector while auto WB is on fights the ISP.
        if self.checks["wb_auto"]:
            return
        red = self.camera.get_balance_ratio("Red")
        if red is not None:
            self._cached_red = red
            self._safe_set_pos(
                "WB Red",
                self._lin_to_pos(red, *self._wb_range, self.WB_STEPS),
            )
        blue = self.camera.get_balance_ratio("Blue")
        if blue is not None:
            self._cached_blue = blue
            self._safe_set_pos(
                "WB Blue",
                self._lin_to_pos(blue, *self._wb_range, self.WB_STEPS),
            )

    def _mouse_to_panel(self, x: int, y: int) -> tuple[int, int]:
        ix, iy = int(x), int(y)
        if 0 <= ix < self.PANEL_W and 0 <= iy < self.PANEL_H:
            return ix, iy
        try:
            _wx, _wy, ww, wh = cv2.getWindowImageRect(self.win)
            ww, wh = int(ww), int(wh)
        except Exception:
            return ix, iy
        if ww < 8 or wh < 8:
            return ix, iy
        cx = int(round(ix * self.PANEL_W / float(ww)))
        cy = int(round(iy * self.PANEL_H / float(wh)))
        return cx, cy

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            return
        if event == cv2.EVENT_LBUTTONUP and getattr(self, "_check_clicked", False):
            self._check_clicked = False
            return
        cx, cy = self._mouse_to_panel(x, y)
        for key, x1, y1, x2, y2 in self._hitboxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                self.checks[key] = not bool(self.checks[key])
                self._check_clicked = event == cv2.EVENT_LBUTTONDOWN
                self._last_render = 0.0
                self.render()
                return

    def sync_quality(self) -> bool:
        """Apply HQ Grab checkbox. Returns True if the mode actually changed."""
        high = bool(self.checks["hq_grab"])
        if high == self._quality_high:
            return False
        self._quality_high = high
        apply_quality_preset(high)
        self.camera.set_quality_mode(high)
        if self.camera.backend == "opencv":
            w, h = _opencv_capture_size(high=high)
            print(f"Quality: {'HQ' if high else 'FAST'} grab {w}x{h}")
        else:
            print(
                "Quality: HQ (1920 / YOLO 1280)"
                if high
                else "Quality: FAST (960 / YOLO 640)"
            )
        return True

    def sync_color_corr(self) -> None:
        on = bool(self.checks["color_corr"])
        if on == self._color_corr:
            return
        self._color_corr = on
        self.camera.set_color_correction(on)
        print("Color correction: ON" if on else "Color correction: OFF")

    def apply(self) -> bool:
        changed = self.sync_quality()
        self.sync_color_corr()
        if self.camera.backend != "spinnaker":
            return changed
        exp_auto = bool(self.checks["exp_auto"])
        gain_auto = bool(self.checks["gain_auto"])
        wb_auto = bool(self.checks["wb_auto"])
        prev = self._prev_autos

        if prev.get("exp_auto") != exp_auto:
            print("Exp Auto: ON" if exp_auto else "Exp Auto: OFF (use Exposure slider)")
        self.camera.set_enum("ExposureAuto", "Continuous" if exp_auto else "Off")
        if not exp_auto:
            exp_pos = cv2.getTrackbarPos("Exposure", self.win)
            force = prev.get("exp_auto", False)
            if force or self._last_manual.get("exp_pos") != exp_pos:
                us = _pos_to_log(exp_pos, *self._exp_range, self.EXP_STEPS)
                if self.camera.set_float("ExposureTime", us):
                    self._cached_exp = us
                    self._last_manual["exp_pos"] = exp_pos
        else:
            self._last_manual.pop("exp_pos", None)

        if prev.get("gain_auto") != gain_auto:
            print("Gain Auto: ON" if gain_auto else "Gain Auto: OFF (use Gain slider)")
        self.camera.set_enum("GainAuto", "Continuous" if gain_auto else "Off")
        if not gain_auto:
            gain_pos = cv2.getTrackbarPos("Gain", self.win)
            force = prev.get("gain_auto", False)
            if force or self._last_manual.get("gain_pos") != gain_pos:
                gain = self._pos_to_lin(gain_pos, *self._gain_range, self.GAIN_STEPS)
                if self.camera.set_float("Gain", gain):
                    self._cached_gain = gain
                    self._last_manual["gain_pos"] = gain_pos
        else:
            self._last_manual.pop("gain_pos", None)

        if prev.get("wb_auto") != wb_auto:
            print("WB Auto: ON" if wb_auto else "WB Auto: OFF (use WB sliders)")
        if wb_auto:
            self.camera.set_enum("BalanceWhiteAuto", "Continuous")
            self._last_manual.pop("wb_red", None)
            self._last_manual.pop("wb_blue", None)
        else:
            self.camera.set_enum("BalanceWhiteAuto", "Off")
            red_pos = cv2.getTrackbarPos("WB Red", self.win)
            blue_pos = cv2.getTrackbarPos("WB Blue", self.win)
            force = prev.get("wb_auto", False)
            if force or self._last_manual.get("wb_red") != red_pos:
                red = self._pos_to_lin(red_pos, *self._wb_range, self.WB_STEPS)
                if self.camera.set_balance_ratio("Red", red):
                    self._cached_red = red
                    self._last_manual["wb_red"] = red_pos
            if force or self._last_manual.get("wb_blue") != blue_pos:
                blue = self._pos_to_lin(blue_pos, *self._wb_range, self.WB_STEPS)
                if self.camera.set_balance_ratio("Blue", blue):
                    self._cached_blue = blue
                    self._last_manual["wb_blue"] = blue_pos

        self._prev_autos = {
            "exp_auto": exp_auto,
            "gain_auto": gain_auto,
            "wb_auto": wb_auto,
        }

        now = time.perf_counter()
        refresh_s = 1.0 if getattr(self.camera, "is_gige", False) else 0.5
        if now - self._last_auto_refresh >= refresh_s:
            self._last_auto_refresh = now
            if exp_auto or gain_auto:
                self._pull_values_from_camera()
        return changed

    def current_readout(self) -> dict[str, str]:
        fps = self.target_fps()
        if self.camera.backend != "spinnaker":
            return {
                "FPS": str(fps),
                "camera": "opencv (no Spinnaker sliders)",
            }
        exp = self._cached_exp
        gain = self._cached_gain
        red = self._cached_red
        blue = self._cached_blue
        return {
            "FPS": str(fps),
            "Exposure": (
                f"{'AUTO' if self.checks['exp_auto'] else 'MAN'}  "
                f"{exp:.0f} us" if exp is not None else "n/a"
            ),
            "Gain": (
                f"{'AUTO' if self.checks['gain_auto'] else 'MAN'}  "
                f"{gain:.2f} dB" if gain is not None else "n/a"
            ),
            "WB": (
                f"{'AUTO' if self.checks['wb_auto'] else 'MAN'}  "
                f"R {red:.2f}  B {blue:.2f}"
                if red is not None and blue is not None
                else ("AUTO" if self.checks["wb_auto"] else "n/a")
            ),
        }

    def _draw_checkbox(self, canvas: np.ndarray, x: int, y: int, key: str, label: str) -> None:
        box = self.BOX
        on = bool(self.checks[key])
        cv2.rectangle(canvas, (x, y), (x + box, y + box), (210, 210, 210), 1)
        if on:
            cv2.rectangle(
                canvas,
                (x + 3, y + 3),
                (x + box - 3, y + box - 3),
                (70, 210, 90),
                thickness=-1,
            )
        cv2.putText(
            canvas,
            label,
            (x + box + 8, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        self._hitboxes.append((key, 8, y - 6, self.PANEL_W - 8, y + box + 8))

    def render(self) -> None:
        now = time.perf_counter()
        if now - getattr(self, "_last_render", 0.0) < 0.25:
            return
        self._last_render = now
        canvas = np.full((self.PANEL_H, self.PANEL_W, 3), 30, dtype=np.uint8)
        self._hitboxes = []
        row_y = 16
        row_h = 32
        for i, (key, label) in enumerate(self.CHECKS):
            self._draw_checkbox(canvas, 16, row_y + i * row_h, key, label)
        y = row_y + row_h * len(self.CHECKS) + 20
        for label, value in self.current_readout().items():
            cv2.putText(
                canvas,
                f"{label}: {value}",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 24
        cv2.putText(
            canvas,
            "Click a box to toggle. Uncheck Auto to use the sliders.",
            (16, self.PANEL_H - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.win, canvas)


def resolve_capture_backend() -> str:
    """Use the configured backend, or fall back to VIDEO_PATH if no camera is present."""
    backend = str(globals().get("CAMERA_BACKEND", "opencv")).lower()
    video_path = Path(globals().get("VIDEO_PATH", "") or "")
    if backend == "opencv":
        return "opencv"
    if backend == "video":
        return "video"
    if backend != "spinnaker":
        return backend
    if list_spinnaker_cameras():
        return "spinnaker"
    if video_path.is_file():
        print(f"No Spinnaker camera; playing video:\n  {video_path}")
        globals()["CAMERA_BACKEND"] = "video"
        return "video"
    raise RuntimeError(
        "No Spinnaker cameras detected. Set CAMERA_BACKEND='opencv' for the USB "
        "camera, or CAMERA_BACKEND='video' with VIDEO_PATH."
    )


def run_live_view() -> None:
    """Blocking OpenCV loop. Press q or Esc in the window to stop."""
    global model, class_names, resolved_device
    if model is None:
        load_model()

    backend = resolve_capture_backend()
    camera = LiveCamera(
        backend,
        index=CAMERA_INDEX,
        spinnaker_index=SPINNAKER_CAMERA_INDEX,
        video_path=globals().get("VIDEO_PATH"),
    )
    camera.open()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    controls = CameraControlPanel(camera, TARGET_FPS)
    view = {"w": 1, "h": 1, "scale": 1.0, "ox": 0.0, "oy": 0.0}
    roi_state = {"pt1": None, "pt2": None, "dragging": False, "box": None}

    def on_mouse(event, x, y, _flags, _param) -> None:
        ix, iy = _mouse_to_image(x, y, view)
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_state["dragging"] = True
            roi_state["pt1"] = (ix, iy)
            roi_state["pt2"] = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and roi_state["dragging"]:
            roi_state["pt2"] = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and roi_state["dragging"]:
            roi_state["dragging"] = False
            roi_state["pt2"] = (ix, iy)
            box = _normalize_roi(
                (*roi_state["pt1"], *roi_state["pt2"]),
                view["w"],
                view["h"],
            )
            roi_state["box"] = box
            if box is None:
                roi_state["pt1"] = roi_state["pt2"] = None

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    t0 = time.perf_counter()
    frames = 0

    print(
        f"Live view started — sliders in '{CONTROLS_WINDOW_NAME}', "
        f"drag ROI on '{WINDOW_NAME}', press q/Esc to stop."
    )

    try:
        while True:
            loop_start = time.perf_counter()
            if controls.apply():
                roi_state["box"] = None
                roi_state["pt1"] = roi_state["pt2"] = None
                roi_state["dragging"] = False
                view["logged"] = False
            target_fps = controls.target_fps()
            period = 1.0 / float(target_fps)

            rgb = camera.read_rgb()
            grab_ms = float(getattr(camera, "last_grab_ms", 0.0))
            if rgb is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            frames += 1
            h, w = rgb.shape[:2]
            view["w"], view["h"] = w, h
            if not view.get("logged"):
                native = getattr(camera, "last_native_hw", None)
                native_s = f"{native[0]}x{native[1]}" if native else "?"
                backend = str(globals().get("MODEL_BACKEND", "yolo"))
                infer_note = (
                    f"cascade imgsz={int(cascade_image_size)}"
                    if backend == "cascade"
                    else f"YOLO imgsz={int(globals().get('INFER_IMGSZ', 640))}"
                )
                print(
                    f"Camera frame: {native_s}  working: {w}x{h}  "
                    f"({infer_note}, device={resolved_device})"
                )
                view["logged"] = True

            t_infer = time.perf_counter()
            display_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            backend = str(globals().get("MODEL_BACKEND", "yolo")).lower()
            label_map = None
            results = None
            if backend == "cascade":
                label_map = predict_cascade_label_map(rgb)
            else:
                infer_imgsz = int(globals().get("INFER_IMGSZ", 640))
                results = model.predict(
                    display_bgr,
                    imgsz=infer_imgsz,
                    conf=CONF,
                    device=resolved_device,
                    retina_masks=not bool(globals().get("FAST_LIVE", False)),
                    quantize=16 if str(resolved_device).lower() not in {"cpu"} else None,
                    verbose=False,
                    save=False,
                )
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            t_mask = time.perf_counter()
            live_roi = roi_state["box"]
            if roi_state["dragging"] and roi_state["pt1"] and roi_state["pt2"]:
                live_roi = _normalize_roi(
                    (*roi_state["pt1"], *roi_state["pt2"]),
                    w,
                    h,
                )
            if backend == "cascade":
                display_bgr = overlay_label_map(
                    display_bgr,
                    label_map,
                    class_names,
                    CLASS_COLORS_BGR,
                    MASK_ALPHA,
                    roi=live_roi,
                )
                stats = compute_shell_stats_from_label_map(
                    label_map, h, w, roi=live_roi
                )
                draw_stats_hud(display_bgr, stats)
            elif results:
                display_bgr = overlay_masks(
                    display_bgr,
                    results[0],
                    class_names,
                    CLASS_COLORS_BGR,
                    MASK_ALPHA,
                    roi=live_roi,
                )
                stats = compute_shell_stats(
                    results[0], class_names, h, w, roi=live_roi
                )
                draw_stats_hud(display_bgr, stats)
            else:
                draw_stats_hud(display_bgr, compute_shell_stats(None, class_names, h, w))
            if live_roi is not None:
                x1, y1, x2, y2 = live_roi
                cv2.rectangle(display_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            draw_legend(display_bgr, class_names)
            mask_ms = (time.perf_counter() - t_mask) * 1000.0
            if not view.get("window_ready"):
                init_w = min(int(globals().get("DISPLAY_MAX_WIDTH", 960)), w)
                init_h = max(1, int(round(init_w * h / float(w))))
                cv2.resizeWindow(WINDOW_NAME, init_w, init_h)
                view["disp_w"], view["disp_h"] = init_w, init_h
                view["window_ready"] = True
            disp_w, disp_h = _video_window_size(WINDOW_NAME, view, w, h)
            shown, scale, ox, oy = _letterbox_to_size(display_bgr, disp_w, disp_h)
            view["scale"] = scale
            view["ox"] = ox
            view["oy"] = oy
            view["disp_w"], view["disp_h"] = shown.shape[1], shown.shape[0]
            work_s = max(1e-6, time.perf_counter() - loop_start)
            pipe_fps = 1.0 / work_s
            view["fps"] = 0.8 * float(view.get("fps") or pipe_fps) + 0.2 * pipe_fps
            view["fps_max"] = max(float(view.get("fps_max") or 0.0), pipe_fps)
            device_text = (
                f"CUDA:{resolved_device}"
                if str(resolved_device).lower() not in {"cpu"}
                else "CPU"
            )
            show_ms = max(0.0, work_s * 1000.0 - grab_ms - infer_ms - mask_ms)
            stages = {
                "grab": grab_ms,
                "infer": infer_ms,
                "mask": mask_ms,
                "show": show_ms,
            }
            for key, val in stages.items():
                ema_key = f"ema_{key}"
                view[ema_key] = 0.8 * float(view.get(ema_key) or val) + 0.2 * val
            limit = max(stages, key=lambda k: float(view.get(f"ema_{k}") or 0.0))
            extra = [
                (
                    "cascade"
                    if str(globals().get("MODEL_BACKEND", "yolo")).lower() == "cascade"
                    else ("HQ" if not bool(globals().get("FAST_LIVE", False)) else "fast")
                ),
                f"grab {view['ema_grab']:.0f}ms",
                f"infer {view['ema_infer']:.0f}ms",
                f"mask {view['ema_mask']:.0f}ms",
                f"limit: {limit}",
            ]
            draw_fps_hud(shown, view["fps"], view["fps_max"], device_text, extra)
            controls.render()

            elapsed = time.perf_counter() - t0
            if elapsed >= 2.0:
                fps = frames / elapsed
                cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} — {fps:.1f} fps")
                frames = 0
                t0 = time.perf_counter()

            cv2.imshow(WINDOW_NAME, shown)
            remaining_ms = int((period - (time.perf_counter() - loop_start)) * 1000)
            key = cv2.waitKey(max(1, remaining_ms)) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("c"), ord("C")):
                roi_state["pt1"] = roi_state["pt2"] = roi_state["box"] = None
                roi_state["dragging"] = False
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Stopped.")




def main() -> None:
    parser = argparse.ArgumentParser(description="Live camera / video YOLO-seg overlay")
    parser.add_argument(
        "--video",
        default=None,
        help="Play this video instead of the camera (loops until q/Esc).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Checkpoint under weights/ or an absolute .pt path (skips the picker).",
    )
    parser.add_argument(
        "--list-weights",
        action="store_true",
        help="Print available weights/ checkpoints and exit.",
    )
    parser.add_argument(
        "--no-choose-weights",
        action="store_true",
        help="Skip the interactive model picker (use default / --weights / env).",
    )
    args = parser.parse_args()
    global CAMERA_BACKEND, VIDEO_PATH

    if args.list_weights:
        models = list_available_weights()
        if not models:
            print(f"No .pt files in {WEIGHTS_DIR}")
            raise SystemExit(1)
        for path in models:
            mb = path.stat().st_size / (1024 * 1024)
            print(f"{path.name}\t{mb:.1f} MB")
        return

    if args.video:
        CAMERA_BACKEND = "video"
        VIDEO_PATH = Path(args.video)

    if args.weights:
        set_weights(args.weights)
    elif os.environ.get("PECAN_YOLO_WEIGHTS"):
        set_weights(os.environ["PECAN_YOLO_WEIGHTS"])
    elif not args.no_choose_weights and sys.stdin.isatty():
        choose_weights_interactive()

    prepare()
    load_model()
    run_live_view()


if __name__ == "__main__":
    main()

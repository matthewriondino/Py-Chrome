import os
import sys
from dataclasses import dataclass, field

import numpy as np

if getattr(sys, "frozen", False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# FIXED MAX PREVIEW SIZES
# ----------------------------
MAX_PREVIEW_W, MAX_PREVIEW_H = 800, 500
MAX_CH_W, MAX_CH_H = MAX_PREVIEW_W // 4, MAX_PREVIEW_H // 4

# Histogram size (bins x px height)
HIST_W, HIST_H = 256, 120

CHANNEL_PREVIEWS = ["Original", "IR", "R", "G"]

# Scatter sampling cap (to keep plots responsive)
MAX_SCATTER_POINTS = 30000

# ----------------------------
# Presets folder (auto-created)
# ----------------------------
PRESETS_DIR = os.path.join(SCRIPT_DIR, "presets")
os.makedirs(PRESETS_DIR, exist_ok=True)

# ----------------------------
# Default slider values (used for Reset and to define which keys we save)
# ----------------------------
DEFAULT_PRESET = {
    "wb_temp": 6500,   # Kelvin
    "wb_tint": 0,      # -100..100
    "fracRx": 0.7,
    "fracGx": 0.7,
    "fracBY": 1.0,
    "gammaRx": 1.0,
    "gammaRy": 1.0,
    "gammaGx": 1.0,
    "gammaGy": 1.0,
    "gammaBY": 1.0,
    "exposure": 1.0,
}

# list of slider tags we persist as a preset
PRESET_SLIDERS = list(DEFAULT_PRESET.keys())

# Default marker size
DEFAULT_MARKER_SIZE = 2

# Save format options
SAVE_FORMAT_TIFF = "TIFF (.tif)"
SAVE_FORMAT_JPEG = "JPEG (.jpg)"
SAVE_FORMAT_PNG = "PNG (.png)"
SAVE_FORMAT_BMP = "BMP (.bmp)"
SAVE_FORMAT_SPECS = {
    SAVE_FORMAT_TIFF: {"ext": ".tif", "pil_format": "TIFF"},
    SAVE_FORMAT_JPEG: {"ext": ".jpg", "pil_format": "JPEG"},
    SAVE_FORMAT_PNG: {"ext": ".png", "pil_format": "PNG"},
    SAVE_FORMAT_BMP: {"ext": ".bmp", "pil_format": "BMP"},
}


@dataclass
class AppState:
    full_img: np.ndarray | None = None
    preview_img: np.ndarray | None = None
    wb_dropper_active: bool = False

    scatter_theme_ids: dict[str, int | None] = field(
        default_factory=lambda: {"rg": None, "rb": None, "gb": None}
    )

    # Deduplicate repeated runtime errors from analysis widgets
    last_analysis_update_error: str | None = None
    last_analysis_label_mode: bool | None = None

    # Cache scatter sample indices to avoid re-randomizing every slider move.
    scatter_cache_n: int | None = None
    scatter_cache_cap: int | None = None
    scatter_cache_idx: np.ndarray | None = None
    scatter_rng: np.random.Generator = field(default_factory=np.random.default_rng)

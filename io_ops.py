import os

import numpy as np
from PIL import Image

from processing import clamp
from state import SAVE_FORMAT_JPEG, SAVE_FORMAT_TIFF

try:
    import tifffile

    TIFFFILE_AVAILABLE = True
except Exception:
    TIFFFILE_AVAILABLE = False


def is_tiff_path(path):
    ext = os.path.splitext(path or "")[1].lower()
    return ext in (".tif", ".tiff")


def save_format_items_for_input(path, save_format_jpeg, save_format_png, save_format_bmp, save_format_tiff):
    items = [save_format_jpeg, save_format_png, save_format_bmp]
    if is_tiff_path(path):
        items.insert(0, save_format_tiff)
    return items


def _normalize_image_array_to_rgb_float(image_array):
    """
    Convert a loaded image array to float32 RGB in [0, 1] while preserving source precision.
    """
    arr = np.asarray(image_array)

    # Handle channel-first arrays (C, H, W) often found in TIFF stacks.
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = np.moveaxis(arr, 0, 2)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype.kind in "ui":
        denom = float(np.iinfo(arr.dtype).max)
        out = arr.astype(np.float32) / max(denom, 1.0)
    elif arr.dtype.kind == "f":
        out = arr.astype(np.float32)
        finite = np.isfinite(out)
        if not np.any(finite):
            out = np.zeros_like(out, dtype=np.float32)
        else:
            mn = float(np.nanmin(out))
            mx = float(np.nanmax(out))
            if mn >= 0.0 and mx <= 1.0:
                pass
            elif mn >= 0.0 and mx <= 255.0:
                out /= 255.0
            elif mn >= 0.0 and mx <= 65535.0:
                out /= 65535.0
            elif mx > mn:
                print("Normalized floating-point image using min/max due to unexpected range.")
                out = (out - mn) / (mx - mn)
            else:
                out = np.zeros_like(out, dtype=np.float32)
    else:
        out = arr.astype(np.float32)
        mn = float(np.nanmin(out)) if out.size else 0.0
        mx = float(np.nanmax(out)) if out.size else 0.0
        if mx > mn:
            out = (out - mn) / (mx - mn)
        else:
            out = np.zeros_like(out, dtype=np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return clamp(out)


def _load_raw_image_array(path):
    """
    Load image data from disk with TIFF precision preserved when available.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff") and TIFFFILE_AVAILABLE:
        try:
            arr = tifffile.imread(path)
            print(f"Loaded TIFF with tifffile: dtype={arr.dtype}, shape={arr.shape}")
            return arr
        except Exception as e:
            print("tifffile load failed, falling back to PIL:", e)

    with Image.open(path) as im:
        if im.mode in ("RGB", "RGBA", "L", "I;16", "I;16B", "I;16L", "I", "F"):
            return np.asarray(im)
        return np.asarray(im.convert("RGB"))


def load_image_arrays(path, max_preview_w, max_preview_h):
    if not path:
        raise ValueError("No path given")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = _load_raw_image_array(path)
    full_img = _normalize_image_array_to_rgb_float(raw)

    ih, iw = full_img.shape[:2]
    scale = min(max_preview_w / iw, max_preview_h / ih, 1.0)
    pw = max(1, int(round(iw * scale)))
    ph = max(1, int(round(ih * scale)))
    if pw == iw and ph == ih:
        preview_img = full_img.copy()
    else:
        preview_u8 = np.round(full_img * 255.0).astype(np.uint8)
        im_preview = Image.fromarray(preview_u8, mode="RGB").resize((pw, ph), Image.LANCZOS)
        preview_img = np.asarray(im_preview, dtype=np.float32) / 255.0

    return full_img, preview_img


def save_output_array(out, out_path, format_label):
    saved = False
    if format_label == SAVE_FORMAT_TIFF:
        arr16 = np.round(out * 65535.0).astype(np.uint16)
        if TIFFFILE_AVAILABLE:
            try:
                tifffile.imwrite(out_path, arr16, photometric="rgb", dtype=np.uint16)
                print("Saved 16-bit TIFF with tifffile:", out_path)
                saved = True
            except Exception as e:
                print("tifffile write error:", e)
                saved = False
        if not saved:
            try:
                arr8 = np.round(out * 255.0).astype(np.uint8)
                Image.fromarray(arr8, mode="RGB").save(out_path, format="TIFF")
                print("Saved 8-bit TIFF (fallback):", out_path)
                saved = True
            except Exception as e:
                print("Fallback TIFF save failed:", e)
                saved = False
    else:
        try:
            arr8 = np.round(out * 255.0).astype(np.uint8)
            img = Image.fromarray(arr8, mode="RGB")
            if format_label == SAVE_FORMAT_JPEG:
                img.save(out_path, format="JPEG", quality=95)
            else:
                # Save format strings for non-TIFF/JPEG are provided by the caller.
                pil_fmt = "PNG" if out_path.lower().endswith(".png") else "BMP"
                img.save(out_path, format=pil_fmt)
            print(f"Saved {format_label}:", out_path)
            saved = True
        except Exception as e:
            print(f"Failed to save {format_label}:", e)
            saved = False
    return saved

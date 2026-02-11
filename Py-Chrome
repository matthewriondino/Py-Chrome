# Aerochrome IR Tool - Aspect Correct Preview + original defaults
# Full replacement with fixed-scale colored histogram + visual gain and minimum-pixel draw
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg
import os, subprocess
import json
import math

# Try to import tifffile
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except Exception:
    TIFFFILE_AVAILABLE = False

# ----------------------------
# FIXED MAX PREVIEW SIZES
# ----------------------------
MAX_PREVIEW_W, MAX_PREVIEW_H = 800, 500
MAX_CH_W, MAX_CH_H = MAX_PREVIEW_W // 4, MAX_PREVIEW_H // 4

# Histogram size (bins x px height)
HIST_W, HIST_H = 256, 120

CHANNEL_PREVIEWS = ["Original", "IR", "R", "G"]

full_img = None
preview_img = None

# WB Dropper state
wb_dropper_active = False

# Scatter sampling cap (to keep plots responsive)
MAX_SCATTER_POINTS = 30000

# Theme ids for dynamic scatter themes (kept so we can delete/recreate)
_SCATTER_THEME_IDS = {"rg": None, "rb": None, "gb": None}

# ----------------------------
# Presets folder (auto-created)
# ----------------------------
PRESETS_DIR = os.path.join(os.getcwd(), "presets")
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
    "exposure": 1.0
}

# list of slider tags we persist as a preset
PRESET_SLIDERS = list(DEFAULT_PRESET.keys())

# Default marker size
DEFAULT_MARKER_SIZE = 2

# ----------------------------
# Utilities
# ----------------------------
def clamp(x):
    return np.clip(x, 0.0, 1.0)

def normalize_frac(a, b):
    s = a + b
    if s > 1.0:
        a /= s
        b /= s
    return a, b

def _extract_path_from_file_dialog_appdata(app_data):
    """
    Helper to robustly extract a single path from various app_data shapes DearPyGui may give.
    Returns None if no valid path found.
    """
    if not app_data:
        return None
    if isinstance(app_data, dict):
        file_path = app_data.get("file_path_name") or app_data.get("file_path") or app_data.get("file_name")
        if isinstance(file_path, (list, tuple)) and file_path:
            return file_path[0]
        return file_path
    if isinstance(app_data, (list, tuple)) and app_data:
        return app_data[0]
    if isinstance(app_data, str):
        return app_data
    return None

# ----------------------------
# Kelvin -> RGB (Tanner Helland algorithm)
# returns float RGB in [0,1]
# ----------------------------
def kelvin_to_rgb(kelvin):
    kelvin = float(np.clip(kelvin, 1000.0, 40000.0))
    tmp = kelvin / 100.0
    if tmp <= 66.0:
        red = 255.0
    else:
        red = 329.698727446 * ((tmp - 60.0) ** -0.1332047592)
    if tmp <= 66.0:
        green = 99.4708025861 * math.log(tmp) - 161.1195681661
    else:
        green = 288.1221695283 * ((tmp - 60.0) ** -0.0755148492)
    if tmp >= 66.0:
        blue = 255.0
    elif tmp <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(tmp - 10.0) - 305.0447927307
    def _cl(v):
        return float(np.clip(v, 0.0, 255.0) / 255.0)
    return np.array([_cl(red), _cl(green), _cl(blue)], dtype=np.float32)

# ----------------------------
# Apply white balance (Temperature + Tint)
# ----------------------------
NEUTRAL_KELVIN = 6500.0
def apply_white_balance(img, temp_kelvin, tint_value):
    """
    img: HxWx3 float in [0,1]
    returns WB-adjusted image in [0,1], same dtype
    """
    if img is None:
        return None
    src_rgb = kelvin_to_rgb(temp_kelvin)
    ref_rgb = kelvin_to_rgb(NEUTRAL_KELVIN)
    eps = 1e-8
    gains = ref_rgb / (src_rgb + eps)
    tint_norm = float(np.clip(tint_value, -100.0, 100.0)) / 100.0
    g_tint_multiplier = 1.0 - 0.15 * tint_norm
    gains[1] *= g_tint_multiplier
    out = img * gains.reshape((1, 1, 3))
    out = clamp(out)
    return out

# ----------------------------
# WB Dropper: find temp/tint from gray sample
# ----------------------------
def find_wb_from_gray_sample(rgb_sample):
    avg = (rgb_sample[0] + rgb_sample[1] + rgb_sample[2]) / 3.0
    if avg < 0.001:
        return 6500, 0
    desired_gains = np.array([avg / (rgb_sample[0] + 1e-8),
                              avg / (rgb_sample[1] + 1e-8),
                              avg / (rgb_sample[2] + 1e-8)], dtype=np.float32)
    ref_rgb = kelvin_to_rgb(NEUTRAL_KELVIN)
    target_rgb = ref_rgb / (desired_gains + 1e-8)
    best_temp = 6500
    min_error = float('inf')
    for temp in range(2000, 12001, 200):
        color = kelvin_to_rgb(temp)
        error = np.linalg.norm(color - target_rgb)
        if error < min_error:
            min_error = error
            best_temp = temp
    for temp in range(max(2000, best_temp - 200), min(12001, best_temp + 200), 20):
        color = kelvin_to_rgb(temp)
        error = np.linalg.norm(color - target_rgb)
        if error < min_error:
            min_error = error
            best_temp = temp
    temp_gains = ref_rgb / (kelvin_to_rgb(best_temp) + 1e-8)
    green_adjustment_needed = desired_gains[1] / (temp_gains[1] + 1e-8)
    tint_norm = (1.0 - green_adjustment_needed) / 0.15
    tint_value = int(np.clip(tint_norm * 100, -100, 100))
    return best_temp, tint_value

def toggle_wb_dropper(sender=None, app_data=None):
    global wb_dropper_active
    wb_dropper_active = not wb_dropper_active
    if wb_dropper_active:
        dpg.configure_item("wb_dropper_btn", label="Click image to set WB...")
        print("WB Dropper active - click on a gray area in the image")
    else:
        dpg.configure_item("wb_dropper_btn", label="Set WB Reference")
        print("WB Dropper deactivated")

def on_preview_image_click(sender, app_data):
    global wb_dropper_active
    if not wb_dropper_active or preview_img is None:
        return
    if not dpg.is_item_hovered("preview_image"):
        return
    try:
        mouse_pos = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min("preview_image")
        rect_max = dpg.get_item_rect_max("preview_image")
        rel_x = mouse_pos[0] - rect_min[0]
        rel_y = mouse_pos[1] - rect_min[1]
        h, w = preview_img.shape[:2]
        scale = min(MAX_PREVIEW_W / w, MAX_PREVIEW_H / h)
        target_w = int(round(w * scale))
        target_h = int(round(h * scale))
        pad_x = (MAX_PREVIEW_W - target_w) // 2
        pad_y = (MAX_PREVIEW_H - target_h) // 2
        img_x = rel_x - pad_x
        img_y = rel_y - pad_y
        if img_x >= 0 and img_x < target_w and img_y >= 0 and img_y < target_h:
            orig_x = int(img_x / scale)
            orig_y = int(img_y / scale)
            orig_x = int(np.clip(orig_x, 0, w - 1))
            orig_y = int(np.clip(orig_y, 0, h - 1))
            sampled_rgb = preview_img[orig_y, orig_x, :]
            print(f"Sampled pixel at ({orig_x}, {orig_y}): RGB = [{sampled_rgb[0]:.3f}, {sampled_rgb[1]:.3f}, {sampled_rgb[2]:.3f}]")
            temp, tint = find_wb_from_gray_sample(sampled_rgb)
            print(f"Calculated WB: Temperature = {temp}K, Tint = {tint}")
            dpg.set_value("wb_temp", temp)
            dpg.set_value("wb_tint", tint)
            update_main_preview()
            dpg.configure_item("wb_dropper_btn", label="Reference Set")
            wb_dropper_active = False
            print("WB reference set successfully")
        else:
            print(f"Click outside image area: rel_x={rel_x:.1f}, rel_y={rel_y:.1f}, img_x={img_x:.1f}, img_y={img_y:.1f}")
    except Exception as e:
        print(f"Error in WB dropper click: {e}")
        import traceback
        traceback.print_exc()

# ----------------------------
# File-open helpers
# ----------------------------
def load_image_from_path(path):
    global full_img, preview_img
    if not path:
        print("No path given")
        return
    if not os.path.exists(path):
        print("File not found:", path)
        return
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        print("Failed to open file:", e)
        return
    full_img = np.asarray(im, dtype=np.float32) / 255.0
    iw, ih = im.size
    scale = min(MAX_PREVIEW_W / iw, MAX_PREVIEW_H / ih, 1.0)
    pw = max(1, int(round(iw * scale)))
    ph = max(1, int(round(ih * scale)))
    im_preview = im.resize((pw, ph), Image.LANCZOS)
    preview_img = np.asarray(im_preview, dtype=np.float32) / 255.0
    try:
        dpg.set_value("file", path)
    except Exception:
        pass
    try:
        rgba_full = render_into_texture(preview_img, MAX_CH_W, MAX_CH_H)
        dpg.set_value("tex_before_Original", rgba_full.flatten().tolist())
    except Exception as e:
        print("Failed to set tex_before_Original:", e)
    try:
        orig_IR = preview_img[:, :, 2]
        orig_R  = preview_img[:, :, 0]
        orig_G  = preview_img[:, :, 1]
        dpg.set_value("tex_before_IR", render_into_texture(orig_IR, MAX_CH_W, MAX_CH_H, mono=True).flatten().tolist())
        dpg.set_value("tex_before_R",  render_into_texture(orig_R,  MAX_CH_W, MAX_CH_H, mono=True).flatten().tolist())
        dpg.set_value("tex_before_G",  render_into_texture(orig_G,  MAX_CH_W, MAX_CH_H, mono=True).flatten().tolist())
    except Exception as e:
        print("Failed to set before-channel textures:", e)
    update_main_preview()

def open_file_callback(sender, app_data):
    file_path = _extract_path_from_file_dialog_appdata(app_data)
    if not file_path:
        print("No file selected")
        return
    load_image_from_path(file_path)

# ----------------------------
# Preset helpers
# ----------------------------
def _preset_filename_from_name(name):
    safe = "".join(c for c in name if c.isalnum() or c in "-_. ").strip()
    if not safe:
        raise ValueError("Invalid preset name")
    return os.path.join(PRESETS_DIR, safe + ".json")

def list_presets_on_disk():
    items = []
    try:
        for fname in sorted(os.listdir(PRESETS_DIR)):
            if fname.lower().endswith(".json"):
                items.append(os.path.splitext(fname)[0])
    except Exception:
        pass
    return items

def refresh_presets_dropdown():
    items = list_presets_on_disk()
    try:
        dpg.configure_item("preset_combo", items=items)
        if items:
            current = dpg.get_value("preset_combo")
            if not current or current not in items:
                dpg.set_value("preset_combo", items[0])
        else:
            dpg.set_value("preset_combo", "")
    except Exception:
        pass

def save_preset_to_folder(name):
    if not name or not name.strip():
        print("Preset name empty.")
        return False
    try:
        path = _preset_filename_from_name(name)
    except ValueError:
        print("Preset name invalid. Use alphanumerics, -, _, . and spaces.")
        return False
    data = {}
    for tag in PRESET_SLIDERS:
        try:
            data[tag] = dpg.get_value(tag)
        except Exception:
            data[tag] = None
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Preset saved: {path}")
        refresh_presets_dropdown()
        dpg.set_value("preset_combo", os.path.splitext(os.path.basename(path))[0])
        return True
    except Exception as e:
        print("Failed to save preset:", e)
        return False

def load_preset_from_folder(name):
    if not name:
        print("No preset name selected.")
        return False
    path = os.path.join(PRESETS_DIR, name + ".json")
    if not os.path.exists(path):
        print("Preset file missing:", path)
        refresh_presets_dropdown()
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Failed to read preset:", e)
        return False
    changed = False
    for tag in PRESET_SLIDERS:
        if tag in data:
            try:
                dpg.set_value(tag, data[tag])
                changed = True
            except Exception:
                pass
    if changed:
        update_main_preview()
        print(f"Preset loaded: {path}")
        return True
    else:
        print("Preset had no known keys.")
        return False

def delete_selected_preset(name):
    if not name:
        print("No preset selected to delete.")
        return False
    path = os.path.join(PRESETS_DIR, name + ".json")
    if not os.path.exists(path):
        print("Preset file not found to delete:", path)
        refresh_presets_dropdown()
        return False
    try:
        os.remove(path)
        print("Deleted preset:", path)
        refresh_presets_dropdown()
        return True
    except Exception as e:
        print("Failed to delete preset:", e)
        return False

def show_delete_preset_confirm(sender=None, app_data=None):
    sel = dpg.get_value("preset_combo")
    if not sel:
        print("No preset selected to delete.")
        return
    dpg.set_value("preset_to_delete", sel)
    dpg.show_item("delete_preset_modal")

def confirm_delete_preset(sender=None, app_data=None):
    name = dpg.get_value("preset_to_delete")
    if not name:
        dpg.hide_item("delete_preset_modal")
        return
    ok = delete_selected_preset(name)
    dpg.hide_item("delete_preset_modal")
    if ok:
        dpg.configure_item("preset_combo", items=list_presets_on_disk())
        dpg.set_value("preset_to_delete", "")

def reset_to_defaults(sender=None, app_data=None):
    for tag, val in DEFAULT_PRESET.items():
        try:
            dpg.set_value(tag, val)
        except Exception:
            pass
    update_main_preview()
    print("Sliders reset to defaults.")

# ----------------------------
# ASPECT-CORRECT RENDERER
# ----------------------------
def render_into_texture(img, tex_w, tex_h, mono=False):
    if img is None:
        return np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    if img.ndim == 2:
        h, w = img.shape
        src_rgb = np.stack([img, img, img], axis=2)
    else:
        h, w = img.shape[:2]
        src_rgb = img
    scale = min(tex_w / w, tex_h / h)
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    if (h == target_h and w == target_w):
        resized = src_rgb
    else:
        pil = Image.fromarray((src_rgb * 255.0).astype(np.uint8))
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        resized = np.asarray(pil, dtype=np.float32) / 255.0
    canvas = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    x0 = (tex_w - target_w) // 2
    y0 = (tex_h - target_h) // 2
    canvas[y0:y0+target_h, x0:x0+target_w, :3] = resized
    canvas[y0:y0+target_h, x0:x0+target_w, 3] = 1.0
    return canvas

# ----------------------------
# SCIENTIFIC TRANSFORM (unchanged core math)
# ----------------------------
def scientific_irg_transform(img):
    fracRx = dpg.get_value("fracRx")
    fracGx = dpg.get_value("fracGx")
    fracBy = dpg.get_value("fracBY")
    fracRy = 1.0 - fracRx
    fracGy = 1.0 - fracGx
    gammaRx, gammaRy = dpg.get_value("gammaRx"), dpg.get_value("gammaRy")
    gammaGx, gammaGy = dpg.get_value("gammaGx"), dpg.get_value("gammaGy")
    gammaBy = dpg.get_value("gammaBY")
    exposure = dpg.get_value("exposure")
    Z1, Z2, Z3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    eps = 1e-6
    fracBy = max(fracBy, eps)
    fracRx = max(fracRx, eps)
    fracGx = max(fracGx, eps)
    innerY = 1.0 - (Z3 / fracBy)
    innerY = np.clip(innerY, 0.0, 1.0)
    Y = 1.0 - innerY ** (1.0 / gammaBy)
    tmp1 = (1.0 - Y) ** gammaRy
    termR = fracRy * (1.0 - tmp1)
    innerX1 = 1.0 - ((Z1 - termR) / fracRx)
    innerX1 = np.clip(innerX1, 0.0, 1.0)
    X1 = 1.0 - innerX1 ** (1.0 / gammaRx)
    tmp2 = (1.0 - Y) ** gammaGy
    termG = fracGy * (1.0 - tmp2)
    innerX2 = 1.0 - ((Z2 - termG) / fracGx)
    innerX2 = np.clip(innerX2, 0.0, 1.0)
    X2 = 1.0 - innerX2 ** (1.0 / gammaGx)
    out = np.dstack([clamp(Y), clamp(X1), clamp(X2)])
    return clamp(out * exposure)

# ----------------------------
# Warnings & UI updates
# ----------------------------
def update_warnings():
    warnings = []
    if dpg.get_value("fracBY") < 1.0:
        warnings.append("⚠ Blue fraction < 1 → highlights may clip")
    if dpg.get_value("exposure") > 2.0:
        warnings.append("⚠ Exposure high → risk of clipping")
    dpg.configure_item("warning_text", default_value="\n".join(warnings))

def update_channel_previews(out):
    if preview_img is None:
        return
    for ch in CHANNEL_PREVIEWS:
        if ch == "Original":
            try:
                unswapped_display = np.dstack([
                    out[:, :, 1],  # Red = recovered red
                    out[:, :, 2],  # Green = recovered green
                    out[:, :, 0],  # Blue = IR
                ])
                rgba = render_into_texture(unswapped_display, MAX_CH_W, MAX_CH_H)
            except Exception as e:
                print("Failed to build unswapped display for After: Original - falling back:", e)
                rgba = render_into_texture(out, MAX_CH_W, MAX_CH_H)
        elif ch == "IR":
            rgba = render_into_texture(out[:, :, 0], MAX_CH_W, MAX_CH_H, mono=True)
        elif ch == "R":
            rgba = render_into_texture(out[:, :, 1], MAX_CH_W, MAX_CH_H, mono=True)
        elif ch == "G":
            rgba = render_into_texture(out[:, :, 2], MAX_CH_W, MAX_CH_H, mono=True)
        else:
            rgba = np.zeros((MAX_CH_H, MAX_CH_W, 4), dtype=np.float32)
        dpg.set_value(f"tex_{ch}", rgba.flatten().tolist())

# ----------------------------
# Scatter theme builder and dynamic update
# ----------------------------
def _delete_scatter_themes():
    for k in list(_SCATTER_THEME_IDS.keys()):
        tid = _SCATTER_THEME_IDS.get(k)
        if tid is not None:
            try:
                dpg.delete_item(tid)
            except Exception:
                pass
            _SCATTER_THEME_IDS[k] = None

def rebuild_scatter_themes(marker_size):
    _delete_scatter_themes()
    try:
        with dpg.theme() as _t_rg:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(marker_size), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
        _SCATTER_THEME_IDS["rg"] = _t_rg
        dpg.bind_item_theme("series_rg", _t_rg)
    except Exception as e:
        print("Failed to create/bind rg theme:", e)
    try:
        with dpg.theme() as _t_rb:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 128, 0, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(marker_size), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
        _SCATTER_THEME_IDS["rb"] = _t_rb
        dpg.bind_item_theme("series_rb", _t_rb)
    except Exception as e:
        print("Failed to create/bind rb theme:", e)
    try:
        with dpg.theme() as _t_gb:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 0, 255, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(marker_size), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
        _SCATTER_THEME_IDS["gb"] = _t_gb
        dpg.bind_item_theme("series_gb", _t_gb)
    except Exception as e:
        print("Failed to create/bind gb theme:", e)

def on_marker_size_changed(sender, app_data):
    try:
        size = int(app_data)
    except Exception:
        size = dpg.get_value("scatter_marker_size")
    rebuild_scatter_themes(size)
    update_main_preview()

# ----------------------------
# Histogram builder & updater (fixed-scale, robust bincount + visual gain + min pixel)
# ----------------------------
def _hist_counts_bincount(values, bins):
    vals = np.asarray(values).ravel()
    if vals.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    vals = np.clip(vals, 0.0, 1.0)
    inds = np.floor(vals * bins).astype(np.int64)
    inds[inds >= bins] = bins - 1
    counts = np.bincount(inds, minlength=bins)[:bins]
    return counts

def build_histogram_texture(src_img, bins=HIST_W, height=HIST_H):
    """
    Build RGBA float32 image representing the histogram where:
      - IR -> RED
      - Red -> GREEN
      - Green -> BLUE
    Fixed normalization (counts / total pixels). Visual gain and minimum-pixel ensure visibility.
    """
    try:
        if src_img is None:
            return np.zeros((height, bins, 4), dtype=np.float32)

        arr = np.asarray(src_img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return np.zeros((height, bins, 4), dtype=np.float32)

        flat = arr.reshape(-1, arr.shape[2])[:, :3]  # ensure at least 3 channels
        total_pixels = float(flat.shape[0]) if flat.size else 1.0
        if total_pixels <= 0:
            return np.zeros((height, bins, 4), dtype=np.float32)

        # Map channels as requested
        ch_ir    = flat[:, 0]
        ch_red   = flat[:, 1]
        ch_green = flat[:, 2]

        # compute counts robustly
        hist_ir    = _hist_counts_bincount(ch_ir, bins)
        hist_red   = _hist_counts_bincount(ch_red, bins)
        hist_green = _hist_counts_bincount(ch_green, bins)

        # fixed normalization: fraction of pixels per bin
        hir = hist_ir.astype(np.float32) / total_pixels
        hgr = hist_red.astype(np.float32) / total_pixels
        hbr = hist_green.astype(np.float32) / total_pixels

        # visual gain (user adjustable). Default if UI not available = 10.0
        try:
            gain = float(dpg.get_value("hist_gain"))
            if gain <= 0:
                gain = 1.0
        except Exception:
            gain = 10.0

        # Create empty image canvas (height x bins x 4)
        img = np.zeros((height, bins, 4), dtype=np.float32)

        # Draw vertical bars per channel (additive).
        # Use height-1 to avoid off-by-one; ensure non-zero bins draw at least 1 pixel
        for x in range(bins):
            rh = int(round(hir[x] * gain * (height - 1)))
            gh = int(round(hgr[x] * gain * (height - 1)))
            bh = int(round(hbr[x] * gain * (height - 1)))
            # If fraction > 0 but rounding produced 0, force one pixel so it's visible
            if hist_ir[x] > 0 and rh == 0:
                rh = 1
            if hist_red[x] > 0 and gh == 0:
                gh = 1
            if hist_green[x] > 0 and bh == 0:
                bh = 1
            # clamp to available height
            rh = min(rh, height - 1)
            gh = min(gh, height - 1)
            bh = min(bh, height - 1)
            if rh > 0:
                img[height - rh:height, x, 0] = 1.0
            if gh > 0:
                img[height - gh:height, x, 1] = 1.0
            if bh > 0:
                img[height - bh:height, x, 2] = 1.0

        # alpha channel = max color channel per pixel (keeps texture visible)
        img[:, :, 3] = np.max(img[:, :, :3], axis=2)
        return img

    except Exception as e:
        print("build_histogram_texture ERROR:", e)
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        return np.zeros((height, bins, 4), dtype=np.float32)

def update_histogram(src_img):
    try:
        tex = build_histogram_texture(src_img, bins=HIST_W, height=HIST_H)
        dpg.set_value("hist_texture", tex.flatten().tolist())
    except Exception as e:
        print("Failed to update histogram texture:", e)
        import traceback
        traceback.print_exc()
        empty = np.zeros((HIST_H, HIST_W, 4), dtype=np.float32)
        try:
            dpg.set_value("hist_texture", empty.flatten().tolist())
        except Exception:
            pass

# ----------------------------
# Main preview updater
# ----------------------------
def update_main_preview(sender=None, app_data=None):
    if preview_img is None:
        update_histogram(None)
        return
    temp = dpg.get_value("wb_temp")
    tint = dpg.get_value("wb_tint")
    wb_preview = apply_white_balance(preview_img, temp, tint)
    out = scientific_irg_transform(wb_preview)
    rgba = render_into_texture(out, MAX_PREVIEW_W, MAX_PREVIEW_H)
    dpg.set_value("texture", rgba.flatten().tolist())
    update_channel_previews(out)
    update_warnings()
    try:
        use_converted = False
        try:
            use_converted = dpg.get_value("scatter_use_converted")
        except Exception:
            use_converted = False
        src_img = out if use_converted else wb_preview
        if src_img is None:
            update_histogram(None)
            return
        flat = src_img.reshape(-1, 3)
        N = flat.shape[0]
        if N > MAX_SCATTER_POINTS:
            idx = np.random.choice(N, MAX_SCATTER_POINTS, replace=False)
            sample = flat[idx]
        else:
            sample = flat
        r = (sample[:, 0] * 255.0).tolist()
        g = (sample[:, 1] * 255.0).tolist()
        b = (sample[:, 2] * 255.0).tolist()
        dpg.set_value("series_rg", [r, g])
        dpg.set_value("series_rb", [r, b])
        dpg.set_value("series_gb", [g, b])
        update_histogram(src_img)
    except Exception:
        pass

# ----------------------------
# Load / Save (image saving preserved but WB applied before transform)
# ----------------------------
def load_image(sender=None, app_data=None):
    path = dpg.get_value("file")
    load_image_from_path(path)

def save_image(sender=None, app_data=None):
    if full_img is None:
        print("No image loaded.")
        return
    input_path = dpg.get_value("file")
    if input_path:
        folder, name = os.path.split(input_path)
        base, _ = os.path.splitext(name)
        out_path = os.path.join(folder, base + "_IRG.tif")
    else:
        out_path = os.path.join(os.getcwd(), "IRG_output_IRG.tif")
    temp = dpg.get_value("wb_temp")
    tint = dpg.get_value("wb_tint")
    wb_full = apply_white_balance(full_img, temp, tint)
    out = scientific_irg_transform(wb_full)
    arr16 = np.round(out * 65535.0).astype(np.uint16)
    saved = False
    if TIFFFILE_AVAILABLE:
        try:
            tifffile.imwrite(out_path, arr16, photometric='rgb', dtype=np.uint16)
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
            print("Fallback save failed:", e)
            saved = False
    if saved:
        try:
            if os.name == "posix":
                subprocess.run(["open", "-R", out_path])
            elif os.name == "nt":
                subprocess.run(["explorer", "/select,", out_path])
        except Exception:
            pass

# ----------------------------
# UI Setup
# ----------------------------
dpg.create_context()

with dpg.texture_registry():
    dpg.add_dynamic_texture(width=MAX_PREVIEW_W, height=MAX_PREVIEW_H,
                            default_value=[0.0] * (MAX_PREVIEW_W * MAX_PREVIEW_H * 4),
                            tag="texture")
    for ch in CHANNEL_PREVIEWS:
        dpg.add_dynamic_texture(width=MAX_CH_W, height=MAX_CH_H,
                                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                                tag=f"tex_{ch}")
    for ch in CHANNEL_PREVIEWS:
        dpg.add_dynamic_texture(width=MAX_CH_W, height=MAX_CH_H,
                                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                                tag=f"tex_before_{ch}")
    dpg.add_dynamic_texture(width=HIST_W, height=HIST_H,
                            default_value=[0.0] * (HIST_W * HIST_H * 4),
                            tag="hist_texture")

with dpg.window(label="Py-Chrome", width=1300, height=850):

    with dpg.group(horizontal=True):

        # Left panel (controls)
        with dpg.child_window(tag="left_panel",
                              resizable_x=True,
                              autosize_y=True,
                              border=True,
                              horizontal_scrollbar=True,
                              width=500):

            dpg.add_input_text(label="Image Path", tag="file", width=400, readonly=True)

            # Top row: Open + Save IRG Image
            with dpg.group(horizontal=True):
                dpg.add_button(label="Open File...", callback=lambda: dpg.show_item("file_dialog"))
                dpg.add_button(label="Save IRG Image", callback=save_image)

            dpg.add_spacer(height=6)

            # Collapsible Preset section
            with dpg.collapsing_header(label="Presets (folder)", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="Preset name", tag="preset_name_input", width=90, default_value="")
                    dpg.add_button(label="Save Preset to folder", callback=lambda s, a: save_preset_to_folder(dpg.get_value("preset_name_input")))
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    dpg.add_combo(items=[], label="Presets", tag="preset_combo", width=90,
                                  callback=lambda s, a: load_preset_from_folder(dpg.get_value("preset_combo")))
                    dpg.add_button(label="Delete Preset", callback=show_delete_preset_confirm)
                    dpg.add_button(label="Refresh", callback=lambda s,a: refresh_presets_dropdown())

            dpg.add_spacer(height=6)
            dpg.add_button(label="Reset to Default", callback=reset_to_defaults)
            dpg.add_spacer(height=6)

                        # Histogram (outside scatterplots, below them)
            with dpg.collapsing_header(label="Histogram (IR→Red, Red→Green, Green→Blue)", default_open=True):
                dpg.add_text("Histogram shows per-channel frequency. IR mapped to RED, Red mapped to GREEN, Green mapped to BLUE.", wrap=420)
                # Visual gain slider to make small fractions visible (fixed-normalization maintained)
                dpg.add_slider_float(label="Histogram visual gain", tag="hist_gain",
                                     default_value=10.0, min_value=1.0, max_value=200.0,
                                     callback=update_main_preview)
                dpg.add_image("hist_texture", width=HIST_W, height=HIST_H)

            with dpg.collapsing_header(label="White Balance (Temperature + Tint)", default_open=True):
                dpg.add_button(label="Set WB Reference", tag="wb_dropper_btn", callback=toggle_wb_dropper)
                dpg.add_text("Click button, then click a gray area in the image to auto-adjust WB.", wrap=400)
                dpg.add_spacer(height=4)
                dpg.add_slider_int(label="WB Temperature (K)", tag="wb_temp",
                                   default_value=DEFAULT_PRESET["wb_temp"], min_value=2000, max_value=12000,
                                   callback=update_main_preview)
                dpg.add_slider_int(label="WB Tint", tag="wb_tint",
                                   default_value=DEFAULT_PRESET["wb_tint"], min_value=-100, max_value=100,
                                   callback=update_main_preview)
                dpg.add_text("Temperature maps to color temperature (Kelvin). Tint is green<->magenta axis.", wrap=400)

            dpg.add_spacer(height=6)

            with dpg.collapsing_header(label="Fraction sliders", default_open=True):
                for label, tag, default, minv, maxv in [
                    ("Red Vis Fraction", "fracRx", DEFAULT_PRESET["fracRx"], 0.0, 1.0),
                    ("Green Vis Fraction", "fracGx", DEFAULT_PRESET["fracGx"], 0.0, 1.0),
                    ("IR Fraction", "fracBY", DEFAULT_PRESET["fracBY"], 0.0, 1.0)
                ]:
                    dpg.add_slider_float(label=label, tag=tag,
                                         default_value=default, min_value=minv, max_value=maxv,
                                         callback=update_main_preview)

            dpg.add_spacer(height=6)

            with dpg.collapsing_header(label="Gamma & Exposure", default_open=True):
                for label, tag, default, minv, maxv in [
                    ("Gamma Red Visible","gammaRx",DEFAULT_PRESET["gammaRx"],0.1,5.0),
                    ("Gamma Red IR","gammaRy",DEFAULT_PRESET["gammaRy"],0.1,5.0),
                    ("Gamma Green Visible","gammaGx",DEFAULT_PRESET["gammaGx"],0.1,5.0),
                    ("Gamma Green IR","gammaGy",DEFAULT_PRESET["gammaGy"],0.1,5.0),
                    ("Gamma IR","gammaBY",DEFAULT_PRESET["gammaBY"],0.1,5.0),
                    ("Exposure","exposure",DEFAULT_PRESET["exposure"],0.1,5.0)
                ]:
                    dpg.add_slider_float(label=label, tag=tag,
                                         default_value=default, min_value=minv, max_value=maxv,
                                         callback=update_main_preview)

            dpg.add_text("", tag="warning_text", color=(255, 0, 0))
            dpg.add_spacer(height=6)

            # Scatterplots
            with dpg.collapsing_header(label="Scatterplots (IRvR, IRvG, RvG)", default_open=True):
                dpg.add_slider_int(label="Scatter marker size", tag="scatter_marker_size",
                                   default_value=DEFAULT_MARKER_SIZE, min_value=1, max_value=8,
                                   callback=on_marker_size_changed)
                dpg.add_spacer(height=6)
                dpg.add_checkbox(label="Showing After", tag="scatter_use_converted", default_value=True, callback=update_main_preview)
                with dpg.group(horizontal=False):
                    with dpg.plot(label="IR vs R", height=170, width=-1):
                        x_axis_rg = dpg.add_plot_axis(dpg.mvXAxis, label="IR (0-255)")
                        y_axis_rg = dpg.add_plot_axis(dpg.mvYAxis, label="Red (0-255)")
                        dpg.set_axis_limits(x_axis_rg, 0.0, 255.0)
                        dpg.set_axis_limits(y_axis_rg, 0.0, 255.0)
                        dpg.add_scatter_series([], [], parent=y_axis_rg, tag="series_rg")
                    with dpg.plot(label="IR vs G", height=170, width=-1):
                        x_axis_rb = dpg.add_plot_axis(dpg.mvXAxis, label="IR (0-255)")
                        y_axis_rb = dpg.add_plot_axis(dpg.mvYAxis, label="Green (0-255)")
                        dpg.set_axis_limits(x_axis_rb, 0.0, 255.0)
                        dpg.set_axis_limits(y_axis_rb, 0.0, 255.0)
                        dpg.add_scatter_series([], [], parent=y_axis_rb, tag="series_rb")
                    with dpg.plot(label="R vs G", height=170, width=-1):
                        x_axis_gb = dpg.add_plot_axis(dpg.mvXAxis, label="Red (0-255)")
                        y_axis_gb = dpg.add_plot_axis(dpg.mvYAxis, label="Green (0-255)")
                        dpg.set_axis_limits(x_axis_gb, 0.0, 255.0)
                        dpg.set_axis_limits(y_axis_gb, 0.0, 255.0)
                        dpg.add_scatter_series([], [], parent=y_axis_gb, tag="series_gb")

        # file dialog
        with dpg.file_dialog(directory_selector=False,
                             show=False,
                             callback=open_file_callback,
                             tag="file_dialog",
                             width=700,
                             height=400):
            dpg.add_file_extension(".tif")
            dpg.add_file_extension(".tiff")
            dpg.add_file_extension(".JPG")
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")

        # Right panel (previews)
        with dpg.child_window(tag="right_panel", autosize_x=True, autosize_y=True):
            dpg.add_text("Main Preview")
            dpg.add_image("texture", tag="preview_image")
            dpg.add_spacer(height=8)
            dpg.add_text("Before Conversion Original / Channels")
            with dpg.group(horizontal=True):
                for ch in CHANNEL_PREVIEWS:
                    with dpg.group():
                        dpg.add_text(f"Before: {ch}")
                        dpg.add_image(f"tex_before_{ch}")
            dpg.add_separator()
            dpg.add_text("After Conversion Converted Channels")
            with dpg.group(horizontal=True):
                for ch in CHANNEL_PREVIEWS:
                    with dpg.group():
                        dpg.add_text(f"After: {ch}")
                        dpg.add_image(f"tex_{ch}")

# Delete preset confirmation modal
with dpg.window(label="Confirm delete preset", modal=True, show=False, tag="delete_preset_modal", no_title_bar=False, width=400, height=120):
    dpg.add_text("Are you sure you want to permanently delete this preset?")
    dpg.add_spacer(height=6)
    dpg.add_text("", tag="delete_preset_name_display")
    dpg.add_spacer(height=6)
    dpg.add_text("", tag="preset_to_delete", show=False)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Confirm Delete", callback=confirm_delete_preset)
        dpg.add_button(label="Cancel", callback=lambda s,a: dpg.hide_item("delete_preset_modal"))

# Create viewport and show
dpg.create_viewport(title="Py-Chrome", width=1300, height=850)
dpg.setup_dearpygui()

# Add global mouse click handler for WB dropper
with dpg.handler_registry():
    dpg.add_mouse_click_handler(callback=on_preview_image_click)

# populate presets dropdown at start
refresh_presets_dropdown()

# create initial scatter themes and bind them
rebuild_scatter_themes(dpg.get_value("scatter_marker_size") if dpg.does_item_exist("scatter_marker_size") else DEFAULT_MARKER_SIZE)

# initialize the histogram texture empty (correct shape: height x width x 4)
empty_hist = np.zeros((HIST_H, HIST_W, 4), dtype=np.float32)
dpg.set_value("hist_texture", empty_hist.flatten().tolist())

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

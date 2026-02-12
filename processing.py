import math

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

NEUTRAL_KELVIN = 6500.0


def clamp(x):
    return np.clip(x, 0.0, 1.0)


def normalize_frac(a, b):
    s = a + b
    if s > 1.0:
        a /= s
        b /= s
    return a, b


def _as_texture_value(rgba):
    """
    DearPyGui accepts NumPy buffers directly; avoid costly Python list conversion.
    """
    return np.ascontiguousarray(rgba, dtype=np.float32).reshape(-1)


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
    desired_gains = np.array(
        [
            avg / (rgb_sample[0] + 1e-8),
            avg / (rgb_sample[1] + 1e-8),
            avg / (rgb_sample[2] + 1e-8),
        ],
        dtype=np.float32,
    )
    ref_rgb = kelvin_to_rgb(NEUTRAL_KELVIN)
    target_rgb = ref_rgb / (desired_gains + 1e-8)
    best_temp = 6500
    min_error = float("inf")
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
    if h == target_h and w == target_w:
        resized = src_rgb
    else:
        pil = Image.fromarray((src_rgb * 255.0).astype(np.uint8))
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        resized = np.asarray(pil, dtype=np.float32) / 255.0
    canvas = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    x0 = (tex_w - target_w) // 2
    y0 = (tex_h - target_h) // 2
    canvas[y0 : y0 + target_h, x0 : x0 + target_w, :3] = resized
    canvas[y0 : y0 + target_h, x0 : x0 + target_w, 3] = 1.0
    return canvas


def _resize_rgb_for_canvas(src_rgb, tex_w, tex_h):
    h, w = src_rgb.shape[:2]
    scale = min(tex_w / w, tex_h / h)
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    if h == target_h and w == target_w:
        resized = src_rgb
    else:
        pil = Image.fromarray((src_rgb * 255.0).astype(np.uint8), mode="RGB")
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        resized = np.asarray(pil, dtype=np.float32) / 255.0
    x0 = (tex_w - target_w) // 2
    y0 = (tex_h - target_h) // 2
    return resized, x0, y0, target_w, target_h


def _build_channel_texture_pack(
    src_rgb,
    tex_w,
    tex_h,
    original_order=(0, 1, 2),
    mono_channel_order=(0, 1, 2),
):
    """
    Build one RGB preview plus three mono previews with a single resize pass.
    """
    resized, x0, y0, target_w, target_h = _resize_rgb_for_canvas(src_rgb, tex_w, tex_h)
    alpha = np.zeros((tex_h, tex_w), dtype=np.float32)
    alpha[y0 : y0 + target_h, x0 : x0 + target_w] = 1.0

    tex_original = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    tex_original[y0 : y0 + target_h, x0 : x0 + target_w, :3] = resized[:, :, list(original_order)]
    tex_original[:, :, 3] = alpha

    mono_textures = []
    for ch in mono_channel_order:
        tex = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
        mono = resized[:, :, ch]
        tex[y0 : y0 + target_h, x0 : x0 + target_w, :3] = mono[:, :, None]
        tex[:, :, 3] = alpha
        mono_textures.append(tex)
    return tex_original, mono_textures


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


def _hist_counts_bincount(values, bins):
    vals = np.asarray(values).ravel()
    if vals.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    vals = np.clip(vals, 0.0, 1.0)
    inds = np.floor(vals * bins).astype(np.int64)
    inds[inds >= bins] = bins - 1
    counts = np.bincount(inds, minlength=bins)[:bins]
    return counts


def build_histogram_texture(src_img, gain, bins=256, height=120):
    """
    Build RGBA float32 image representing histogram bars where:
      - Channel 0 -> RED
      - Channel 1 -> GREEN
      - Channel 2 -> BLUE
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

        # Map channels by index.
        ch0 = flat[:, 0]
        ch1 = flat[:, 1]
        ch2 = flat[:, 2]

        # compute counts robustly
        hist_0 = _hist_counts_bincount(ch0, bins)
        hist_1 = _hist_counts_bincount(ch1, bins)
        hist_2 = _hist_counts_bincount(ch2, bins)

        # fixed normalization: fraction of pixels per bin
        h0 = hist_0.astype(np.float32) / total_pixels
        h1 = hist_1.astype(np.float32) / total_pixels
        h2 = hist_2.astype(np.float32) / total_pixels

        # visual gain (user adjustable)
        if gain <= 0:
            gain = 1.0

        # Create empty image canvas (height x bins x 4)
        img = np.zeros((height, bins, 4), dtype=np.float32)

        # Vectorized vertical bar drawing per channel.
        rh = np.rint(h0 * gain * (height - 1)).astype(np.int32)
        gh = np.rint(h1 * gain * (height - 1)).astype(np.int32)
        bh = np.rint(h2 * gain * (height - 1)).astype(np.int32)

        # If fraction > 0 but rounding produced 0, force one pixel so it's visible.
        rh[(hist_0 > 0) & (rh == 0)] = 1
        gh[(hist_1 > 0) & (gh == 0)] = 1
        bh[(hist_2 > 0) & (bh == 0)] = 1

        rh = np.clip(rh, 0, height - 1)
        gh = np.clip(gh, 0, height - 1)
        bh = np.clip(bh, 0, height - 1)

        rows = np.arange(height, dtype=np.int32)[:, None]
        img[:, :, 0] = (rows >= (height - rh[None, :])).astype(np.float32)
        img[:, :, 1] = (rows >= (height - gh[None, :])).astype(np.float32)
        img[:, :, 2] = (rows >= (height - bh[None, :])).astype(np.float32)

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


def scatter_pairs_from_sample(sample, use_converted):
    """
    Returns x/y vectors for the three scatter plots in fixed order:
      Plot 1: After Red vs IR / Before Red vs source channel-2 (shown as IR)
      Plot 2: After Green vs IR / Before Green vs source channel-2 (shown as IR)
      Plot 3: Red vs Green (both modes)
    """
    if use_converted:
        x1, y1 = sample[:, 1], sample[:, 0]  # Red vs IR
        x2, y2 = sample[:, 2], sample[:, 0]  # Green vs IR
        x3, y3 = sample[:, 1], sample[:, 2]  # Red vs Green
    else:
        x1, y1 = sample[:, 0], sample[:, 2]  # Red vs source channel-2 (labeled IR)
        x2, y2 = sample[:, 1], sample[:, 2]  # Green vs source channel-2 (labeled IR)
        x3, y3 = sample[:, 0], sample[:, 1]  # Red vs Green
    return x1, y1, x2, y2, x3, y3


def sample_scatter_points(flat, state, max_scatter_points):
    n = int(flat.shape[0])
    cap = int(max_scatter_points)
    if n <= cap:
        return flat
    if state.scatter_cache_idx is None or state.scatter_cache_n != n or state.scatter_cache_cap != cap:
        state.scatter_cache_idx = state.scatter_rng.choice(n, cap, replace=False)
        state.scatter_cache_n = n
        state.scatter_cache_cap = cap
    return flat[state.scatter_cache_idx]

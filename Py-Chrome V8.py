# Aerochrome IR Tool - Aspect Correct Preview + original defaults
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg
import os, subprocess

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

CHANNEL_PREVIEWS = ["Original", "IR", "R", "G"]

full_img = None
preview_img = None

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
# ----------------------------
# File-open helpers (add here)
# ----------------------------
def load_image_from_path(path):
    """Central loader used by file dialog and text-path loader.
    Also populates the BEFORE/Original channel thumbnails.
    """
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

    # Make a downsampled preview that's at most MAX_PREVIEW_W x MAX_PREVIEW_H.
    iw, ih = im.size
    scale = min(MAX_PREVIEW_W / iw, MAX_PREVIEW_H / ih, 1.0)  # do not upscale small images
    pw = max(1, int(round(iw * scale)))
    ph = max(1, int(round(ih * scale)))
    im_preview = im.resize((pw, ph), Image.LANCZOS)
    preview_img = np.asarray(im_preview, dtype=np.float32) / 255.0

    # Update the UI file textbox so path is visible
    dpg.set_value("file", path)

    # --- populate BEFORE/original previews (render into the fixed channel textures) ---
    # full RGB original (letterboxed into channel preview size)
    try:
        rgba_full = render_into_texture(preview_img, MAX_CH_W, MAX_CH_H)
        dpg.set_value("tex_before_Original", rgba_full.flatten().tolist())
    except Exception as e:
        print("Failed to set tex_before_Original:", e)

    # original raw channels (from the preview image)
    try:
        orig_IR = preview_img[:, :, 2]   # blue channel contains IR on full-spectrum+yellow images
        orig_R  = preview_img[:, :, 0]
        orig_G  = preview_img[:, :, 1]

        dpg.set_value("tex_before_IR", render_into_texture(orig_IR, MAX_CH_W, MAX_CH_H, mono=True).flatten().tolist())
        dpg.set_value("tex_before_R",  render_into_texture(orig_R,  MAX_CH_W, MAX_CH_H, mono=True).flatten().tolist())
        dpg.set_value("tex_before_G",  render_into_texture(orig_G,  MAX_CH_W, MAX_CH_H, mono=True).flatten().tolist())
    except Exception as e:
        print("Failed to set before-channel textures:", e)

    # now update the main converted preview + converted channel previews
    update_main_preview()



def open_file_callback(sender, app_data):
    """
    Robust file dialog handler: DPG may pass a dict or list depending on version/platform.
    Extract a single path and call the loader.
    """
    file_path = None
    # some DPG versions pass a dict with file_path_name
    if isinstance(app_data, dict):
        file_path = app_data.get("file_path_name") or app_data.get("file_path") or app_data.get("file_name")
        if isinstance(file_path, (list, tuple)) and file_path:
            file_path = file_path[0]
    elif isinstance(app_data, (list, tuple)) and app_data:
        # older behaviour — a list of selected files
        file_path = app_data[0]

    if not file_path:
        print("No file selected")
        return

    load_image_from_path(file_path)


# ----------------------------
# ASPECT-CORRECT RENDERER
# ----------------------------
# Optimised renderer: avoid PIL resize when unnecessary
def render_into_texture(img, tex_w, tex_h, mono=False):
    """
    img: float HxWx3 (or HxW if mono)
    returns: tex_h x tex_w x 4 float RGBA (0..1)
    This version avoids doing a PIL resize when the source is already
    <= target size; it uses a fast numpy copy for centering.
    """
    if img is None:
        return np.zeros((tex_h, tex_w, 4), dtype=np.float32)

    # If grayscale (2D), expand to RGB for placement
    if img.ndim == 2:
        h, w = img.shape
        src_rgb = np.stack([img, img, img], axis=2)
    else:
        h, w = img.shape[:2]
        src_rgb = img

    # Compute scale to fit inside texture
    scale = min(tex_w / w, tex_h / h)

    # If the source already matches the scaled size (or is smaller), avoid PIL resizing:
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))

    if (h == target_h and w == target_w):
        # no resizing needed, src_rgb can be placed directly (but may be smaller than tex)
        resized = src_rgb
    else:
        # do a single PIL resize when needed
        pil = Image.fromarray((src_rgb * 255.0).astype(np.uint8))
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        resized = np.asarray(pil, dtype=np.float32) / 255.0

    # Compose onto the canvas (centered letterbox)
    canvas = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    x0 = (tex_w - target_w) // 2
    y0 = (tex_h - target_h) // 2
    canvas[y0:y0+target_h, x0:x0+target_w, :3] = resized
    canvas[y0:y0+target_h, x0:x0+target_w, 3] = 1.0
    return canvas

# ----------------------------
# SCIENTIFIC TRANSFORM (JW Wong inverse)
# ----------------------------

def scientific_irg_transform(img):
    # sliders
    fracRx, fracRy = dpg.get_value("fracRx"), dpg.get_value("fracRy")
    fracGx, fracGy = dpg.get_value("fracGx"), dpg.get_value("fracGy")
    fracBy = dpg.get_value("fracBY")

    gammaRx, gammaRy = dpg.get_value("gammaRx"), dpg.get_value("gammaRy")
    gammaGx, gammaGy = dpg.get_value("gammaGx"), dpg.get_value("gammaGy")
    gammaBy = dpg.get_value("gammaBY")

    exposure = dpg.get_value("exposure")

    # normalize fractions to avoid sums > 1
    fracRx, fracRy = normalize_frac(fracRx, fracRy)
    fracGx, fracGy = normalize_frac(fracGx, fracGy)

    Z1, Z2, Z3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    eps = 1e-6
    fracBy = max(fracBy, eps)
    fracRx = max(fracRx, eps)
    fracGx = max(fracGx, eps)

    # Numerically-safe inner values clipped to [0,1]
    innerY = 1.0 - (Z3 / fracBy)
    innerY = np.clip(innerY, 0.0, 1.0)
    Y = 1.0 - innerY ** (1.0 / gammaBy)

    # X1 inner: (Z1 - fracRy*(1 - (1 - Y)**gammaRy)) / fracRx
    tmp1 = (1.0 - Y) ** gammaRy
    termR = fracRy * (1.0 - tmp1)
    innerX1 = 1.0 - ((Z1 - termR) / fracRx)
    innerX1 = np.clip(innerX1, 0.0, 1.0)
    X1 = 1.0 - innerX1 ** (1.0 / gammaRx)

    # X2 inner: similar for green
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
    if dpg.get_value("fracRx") + dpg.get_value("fracRy") > 1.0:
        warnings.append("⚠ Red fractions sum > 1 → negative values possible")
    if dpg.get_value("fracGx") + dpg.get_value("fracGy") > 1.0:
        warnings.append("⚠ Green fractions sum > 1 → negative values possible")
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
            rgba = render_into_texture(preview_img, MAX_CH_W, MAX_CH_H)
        elif ch == "IR":
            rgba = render_into_texture(out[:, :, 0], MAX_CH_W, MAX_CH_H, mono=True)
        elif ch == "R":
            rgba = render_into_texture(out[:, :, 1], MAX_CH_W, MAX_CH_H, mono=True)
        elif ch == "G":
            rgba = render_into_texture(out[:, :, 2], MAX_CH_W, MAX_CH_H, mono=True)
        else:
            rgba = np.zeros((MAX_CH_H, MAX_CH_W, 4), dtype=np.float32)

        dpg.set_value(f"tex_{ch}", rgba.flatten().tolist())

def update_main_preview(sender=None, app_data=None):
    if preview_img is None:
        return

    out = scientific_irg_transform(preview_img)
    rgba = render_into_texture(out, MAX_PREVIEW_W, MAX_PREVIEW_H)
    dpg.set_value("texture", rgba.flatten().tolist())

    update_channel_previews(out)
    update_warnings()

# ----------------------------
# Load / Save
# ----------------------------
# Load image: create a small preview copy and keep full_img for saving
def load_image(sender=None, app_data=None):
    """Keep backward compatibility with the text-box loader."""
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

    out = scientific_irg_transform(full_img)
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
            # reveal in Finder / Explorer if possible
            if os.name == "posix":
                subprocess.run(["open", "-R", out_path])
            elif os.name == "nt":
                subprocess.run(["explorer", "/select,", out_path])
        except Exception:
            pass

# ----------------------------
# UI Setup (with your original defaults)
# ----------------------------
dpg.create_context()

# create textures once (max sizes)
with dpg.texture_registry():
    # main preview texture (max size)
    dpg.add_dynamic_texture(width=MAX_PREVIEW_W, height=MAX_PREVIEW_H,
                            default_value=[0.0] * (MAX_PREVIEW_W * MAX_PREVIEW_H * 4),
                            tag="texture")

    # converted / "after" channel textures (existing)
    for ch in CHANNEL_PREVIEWS:
        dpg.add_dynamic_texture(width=MAX_CH_W, height=MAX_CH_H,
                                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                                tag=f"tex_{ch}")

    # BEFORE / original textures (new): keep separate tags so both sets exist
    for ch in CHANNEL_PREVIEWS:
        dpg.add_dynamic_texture(width=MAX_CH_W, height=MAX_CH_H,
                                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                                tag=f"tex_before_{ch}")


with dpg.window(label="Py-Chrome", width=1300, height=850):

    with dpg.group(horizontal=True):

        # Left panel (controls)
        with dpg.child_window(tag="left_panel",
                              resizable_x=True,
                              autosize_y=True,
                              border=True,
                              horizontal_scrollbar=True,
                              width=500):

            # show selected file path (readonly) so users can copy it if needed
            dpg.add_input_text(label="Image Path", tag="file", width=400, readonly=True)

            # Open file button — shows file dialog (see file_dialog block below)
            dpg.add_button(label="Open File...", callback=lambda: dpg.show_item("file_dialog"))

            # Save uses your existing save_image function
            dpg.add_button(label="Save IRG Image", callback=save_image)

            dpg.add_spacer(height=6)

            # Fraction sliders (unchanged)
            for label, tag, default, minv, maxv in [
                ("Red Fraction (Visible)", "fracRx", 0.7, 0.0, 1.0),
                ("Red Fraction (IR)", "fracRy", 0.3, 0.0, 1.0),
                ("Green Fraction (Visible)", "fracGx", 0.7, 0.0, 1.0),
                ("Green Fraction (IR)", "fracGy", 0.3, 0.0, 1.0),
                ("Blue Fraction (IR only)", "fracBY", 1.0, 0.0, 1.0)
            ]:
                dpg.add_slider_float(label=label, tag=tag,
                                     default_value=default, min_value=minv, max_value=maxv,
                                     callback=update_main_preview)

            dpg.add_spacer(height=6)

            # Gamma + Exposure sliders (unchanged)
            for label, tag, default, minv, maxv in [
                ("Gamma Red Visible","gammaRx",1.0,0.1,5.0),
                ("Gamma Red IR","gammaRy",1.0,0.1,5.0),
                ("Gamma Green Visible","gammaGx",1.0,0.1,5.0),
                ("Gamma Green IR","gammaGy",1.0,0.1,5.0),
                ("Gamma Blue IR","gammaBY",1.0,0.1,5.0),
                ("Exposure","exposure",1.0,0.1,5.0)
            ]:
                dpg.add_slider_float(label=label, tag=tag,
                                     default_value=default, min_value=minv, max_value=maxv,
                                     callback=update_main_preview)

            dpg.add_spacer(height=8)

            # Warning text (updated live by update_warnings)
            dpg.add_text("", tag="warning_text", color=(255, 0, 0))

                    # File dialog (hidden until opened) — placed inside the same window
        with dpg.file_dialog(directory_selector=False,
                             show=False,
                             callback=open_file_callback,
                             tag="file_dialog",
                             width=700,
                             height=400):
            # list common image extensions (no wildcard)
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".bmp")
            dpg.add_file_extension(".tif")
            dpg.add_file_extension(".tiff")

        # Right panel (previews)
        with dpg.child_window(tag="right_panel", autosize_x=True, autosize_y=True):
            # Main editable preview (top)
            dpg.add_text("Main Preview")
            dpg.add_image("texture")

            # BEFORE: Original image + original channels
            dpg.add_spacer(height=8)  # 8 px of vertical space
            dpg.add_text("Before Conversion — Original / Channels")
            with dpg.group(horizontal=True):
                # show original full image and its channels (tex_before_*)
                for ch in CHANNEL_PREVIEWS:
                    with dpg.group():
                        dpg.add_text(f"Before: {ch}")
                        dpg.add_image(f"tex_before_{ch}")

            dpg.add_separator()

            # AFTER: Converted channel previews (existing behaviour)
            dpg.add_text("After Conversion — Converted Channels")
            with dpg.group(horizontal=True):
                for ch in CHANNEL_PREVIEWS:
                    with dpg.group():
                        dpg.add_text(f"After: {ch}")
                        dpg.add_image(f"tex_{ch}")



dpg.create_viewport(title="Py-Chrome", width=1300, height=850)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

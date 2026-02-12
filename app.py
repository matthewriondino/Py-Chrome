import os
import subprocess

import dearpygui.dearpygui as dpg
import numpy as np

import ui
from io_ops import (
    TIFFFILE_AVAILABLE,
    is_tiff_path,
    load_image_arrays,
    save_format_items_for_input,
    save_output_array,
)
from presets import (
    delete_preset_file,
    list_presets_on_disk,
    load_preset_file,
    preset_filename_from_name,
    save_preset_file,
)
from processing import (
    _as_texture_value,
    _build_channel_texture_pack,
    apply_white_balance,
    build_histogram_texture,
    find_wb_from_gray_sample,
    render_into_texture,
    sample_scatter_points,
    scatter_pairs_from_sample,
    scientific_irg_transform,
)
from state import (
    AppState,
    CHANNEL_PREVIEWS,
    DEFAULT_MARKER_SIZE,
    DEFAULT_PRESET,
    HIST_H,
    HIST_W,
    MAX_CH_H,
    MAX_CH_W,
    MAX_PREVIEW_H,
    MAX_PREVIEW_W,
    MAX_SCATTER_POINTS,
    PRESETS_DIR,
    PRESET_SLIDERS,
    SAVE_FORMAT_BMP,
    SAVE_FORMAT_JPEG,
    SAVE_FORMAT_PNG,
    SAVE_FORMAT_SPECS,
    SAVE_FORMAT_TIFF,
)

STATE = AppState()


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


def update_save_format_options(path=None):
    if not dpg.does_item_exist("save_format"):
        return
    items = save_format_items_for_input(
        path,
        SAVE_FORMAT_JPEG,
        SAVE_FORMAT_PNG,
        SAVE_FORMAT_BMP,
        SAVE_FORMAT_TIFF,
    )
    current = dpg.get_value("save_format")
    dpg.configure_item("save_format", items=items)
    if not current or current not in items:
        dpg.set_value("save_format", items[0] if items else "")
    if dpg.does_item_exist("save_format_hint"):
        if is_tiff_path(path):
            dpg.set_value("save_format_hint", "TIFF export available (source file is TIFF).")
        else:
            dpg.set_value("save_format_hint", "TIFF export disabled for non-TIFF source files.")


def update_warnings():
    warnings = []
    if dpg.get_value("fracBY") < 1.0:
        warnings.append("⚠ Blue fraction < 1 → highlights may clip")
    if dpg.get_value("exposure") > 2.0:
        warnings.append("⚠ Exposure high → risk of clipping")
    dpg.configure_item("warning_text", default_value="\n".join(warnings))


def update_analysis_labels(use_converted):
    if use_converted:
        hist_desc = "Histogram shows converted channels. Channel 0 (IR) is drawn in RED, channel 1 (Red) in GREEN, channel 2 (Green) in BLUE."
        label_map = {
            "plot_rg": "Red vs IR",
            "plot_rb": "Green vs IR",
            "plot_gb": "Red vs Green",
            "axis_rg_x": "Red (0-255)",
            "axis_rg_y": "IR (0-255)",
            "axis_rb_x": "Green (0-255)",
            "axis_rb_y": "IR (0-255)",
            "axis_gb_x": "Red (0-255)",
            "axis_gb_y": "Green (0-255)",
        }
    else:
        hist_desc = "Histogram shows source RGB channels. Red is drawn in RED, Green in GREEN, Blue in BLUE."
        label_map = {
            "plot_rg": "Red vs IR",
            "plot_rb": "Green vs IR",
            "plot_gb": "Red vs Green",
            "axis_rg_x": "Red (0-255)",
            "axis_rg_y": "IR (0-255)",
            "axis_rb_x": "Green (0-255)",
            "axis_rb_y": "IR (0-255)",
            "axis_gb_x": "Red (0-255)",
            "axis_gb_y": "Green (0-255)",
        }

    if dpg.does_item_exist("hist_description"):
        dpg.set_value("hist_description", hist_desc)

    for tag, label in label_map.items():
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, label=label)
    STATE.last_analysis_label_mode = use_converted


def _delete_scatter_themes():
    for k in list(STATE.scatter_theme_ids.keys()):
        tid = STATE.scatter_theme_ids.get(k)
        if tid is not None:
            try:
                dpg.delete_item(tid)
            except Exception:
                pass
            STATE.scatter_theme_ids[k] = None


def rebuild_scatter_themes(marker_size):
    _delete_scatter_themes()
    try:
        with dpg.theme() as theme_rg:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(marker_size), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
        STATE.scatter_theme_ids["rg"] = theme_rg
        dpg.bind_item_theme("series_rg", theme_rg)
    except Exception as e:
        print("Failed to create/bind rg theme:", e)

    try:
        with dpg.theme() as theme_rb:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 128, 0, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(marker_size), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
        STATE.scatter_theme_ids["rb"] = theme_rb
        dpg.bind_item_theme("series_rb", theme_rb)
    except Exception as e:
        print("Failed to create/bind rb theme:", e)

    try:
        with dpg.theme() as theme_gb:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 0, 255, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, int(marker_size), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
        STATE.scatter_theme_ids["gb"] = theme_gb
        dpg.bind_item_theme("series_gb", theme_gb)
    except Exception as e:
        print("Failed to create/bind gb theme:", e)


def on_marker_size_changed(sender, app_data):
    try:
        size = int(app_data)
    except Exception:
        size = dpg.get_value("scatter_marker_size")
    rebuild_scatter_themes(size)
    update_main_preview()


def update_channel_previews(out):
    if STATE.preview_img is None:
        return
    try:
        tex_original, mono = _build_channel_texture_pack(
            out,
            MAX_CH_W,
            MAX_CH_H,
            original_order=(1, 2, 0),   # Red, Green, IR
            mono_channel_order=(0, 1, 2),  # IR, R, G
        )
        dpg.set_value("tex_Original", _as_texture_value(tex_original))
        dpg.set_value("tex_IR", _as_texture_value(mono[0]))
        dpg.set_value("tex_R", _as_texture_value(mono[1]))
        dpg.set_value("tex_G", _as_texture_value(mono[2]))
    except Exception as e:
        print("Failed to update channel previews:", e)


def update_histogram(src_img):
    try:
        try:
            gain = float(dpg.get_value("hist_gain"))
            if gain <= 0:
                gain = 1.0
        except Exception:
            gain = 10.0
        tex = build_histogram_texture(src_img, gain=gain, bins=HIST_W, height=HIST_H)
        dpg.set_value("hist_texture", _as_texture_value(tex))
    except Exception as e:
        print("Failed to update histogram texture:", e)
        import traceback

        traceback.print_exc()
        empty = np.zeros((HIST_H, HIST_W, 4), dtype=np.float32)
        try:
            dpg.set_value("hist_texture", _as_texture_value(empty))
        except Exception:
            pass


def update_main_preview(sender=None, app_data=None):
    try:
        use_converted = bool(dpg.get_value("scatter_use_converted"))
    except Exception:
        use_converted = True
    if STATE.last_analysis_label_mode is None or STATE.last_analysis_label_mode != use_converted:
        update_analysis_labels(use_converted)

    if STATE.preview_img is None:
        update_histogram(None)
        return

    temp = dpg.get_value("wb_temp")
    tint = dpg.get_value("wb_tint")
    wb_preview = apply_white_balance(STATE.preview_img, temp, tint)
    out = scientific_irg_transform(wb_preview)
    rgba = render_into_texture(out, MAX_PREVIEW_W, MAX_PREVIEW_H)
    dpg.set_value("texture", _as_texture_value(rgba))
    update_channel_previews(out)
    update_warnings()

    try:
        src_img = out if use_converted else wb_preview
        if src_img is None:
            update_histogram(None)
            return
        flat = src_img.reshape(-1, 3)
        sample = sample_scatter_points(flat, STATE, MAX_SCATTER_POINTS)
        x1, y1, x2, y2, x3, y3 = scatter_pairs_from_sample(sample, use_converted)
        dpg.set_value("series_rg", [np.asarray(x1 * 255.0, dtype=np.float32), np.asarray(y1 * 255.0, dtype=np.float32)])
        dpg.set_value("series_rb", [np.asarray(x2 * 255.0, dtype=np.float32), np.asarray(y2 * 255.0, dtype=np.float32)])
        dpg.set_value("series_gb", [np.asarray(x3 * 255.0, dtype=np.float32), np.asarray(y3 * 255.0, dtype=np.float32)])
        update_histogram(src_img)
        STATE.last_analysis_update_error = None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if msg != STATE.last_analysis_update_error:
            print("Analysis update failed:", msg)
            STATE.last_analysis_update_error = msg


def toggle_wb_dropper(sender=None, app_data=None):
    STATE.wb_dropper_active = not STATE.wb_dropper_active
    if STATE.wb_dropper_active:
        dpg.configure_item("wb_dropper_btn", label="Click image to set WB...")
        print("WB Dropper active - click on a gray area in the image")
    else:
        dpg.configure_item("wb_dropper_btn", label="Set WB Reference")
        print("WB Dropper deactivated")


def on_preview_image_click(sender, app_data):
    if not STATE.wb_dropper_active or STATE.preview_img is None:
        return
    if not dpg.is_item_hovered("preview_image"):
        return

    try:
        mouse_pos = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min("preview_image")
        rel_x = mouse_pos[0] - rect_min[0]
        rel_y = mouse_pos[1] - rect_min[1]
        h, w = STATE.preview_img.shape[:2]
        scale = min(MAX_PREVIEW_W / w, MAX_PREVIEW_H / h)
        target_w = int(round(w * scale))
        target_h = int(round(h * scale))
        pad_x = (MAX_PREVIEW_W - target_w) // 2
        pad_y = (MAX_PREVIEW_H - target_h) // 2
        img_x = rel_x - pad_x
        img_y = rel_y - pad_y
        if 0 <= img_x < target_w and 0 <= img_y < target_h:
            orig_x = int(img_x / scale)
            orig_y = int(img_y / scale)
            orig_x = int(np.clip(orig_x, 0, w - 1))
            orig_y = int(np.clip(orig_y, 0, h - 1))
            sampled_rgb = STATE.preview_img[orig_y, orig_x, :]
            print(
                f"Sampled pixel at ({orig_x}, {orig_y}): RGB = [{sampled_rgb[0]:.3f}, {sampled_rgb[1]:.3f}, {sampled_rgb[2]:.3f}]"
            )
            temp, tint = find_wb_from_gray_sample(sampled_rgb)
            print(f"Calculated WB: Temperature = {temp}K, Tint = {tint}")
            dpg.set_value("wb_temp", temp)
            dpg.set_value("wb_tint", tint)
            update_main_preview()
            dpg.configure_item("wb_dropper_btn", label="Reference Set")
            STATE.wb_dropper_active = False
            print("WB reference set successfully")
        else:
            print(f"Click outside image area: rel_x={rel_x:.1f}, rel_y={rel_y:.1f}, img_x={img_x:.1f}, img_y={img_y:.1f}")
    except Exception as e:
        print(f"Error in WB dropper click: {e}")
        import traceback

        traceback.print_exc()


def load_image_from_path(path):
    if not path:
        print("No path given")
        return
    try:
        full_img, preview_img = load_image_arrays(path, MAX_PREVIEW_W, MAX_PREVIEW_H)
    except FileNotFoundError:
        print("File not found:", path)
        return
    except Exception as e:
        print("Failed to open file:", e)
        return

    STATE.full_img = full_img
    STATE.preview_img = preview_img
    STATE.scatter_cache_idx = None
    STATE.scatter_cache_n = None
    STATE.scatter_cache_cap = None

    try:
        dpg.set_value("file", path)
    except Exception:
        pass

    try:
        update_save_format_options(path)
    except Exception:
        pass

    try:
        tex_before_original, before_mono = _build_channel_texture_pack(
            STATE.preview_img,
            MAX_CH_W,
            MAX_CH_H,
            original_order=(0, 1, 2),
            mono_channel_order=(2, 0, 1),  # IR, R, G
        )
        dpg.set_value("tex_before_Original", _as_texture_value(tex_before_original))
        dpg.set_value("tex_before_IR", _as_texture_value(before_mono[0]))
        dpg.set_value("tex_before_R", _as_texture_value(before_mono[1]))
        dpg.set_value("tex_before_G", _as_texture_value(before_mono[2]))
    except Exception as e:
        print("Failed to set before-channel textures:", e)

    update_main_preview()


def open_file_callback(sender, app_data):
    file_path = _extract_path_from_file_dialog_appdata(app_data)
    if not file_path:
        print("No file selected")
        return
    load_image_from_path(file_path)


def refresh_presets_dropdown():
    items = list_presets_on_disk(PRESETS_DIR)
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


def _write_preset_file(path):
    data = {}
    for tag in PRESET_SLIDERS:
        try:
            data[tag] = dpg.get_value(tag)
        except Exception:
            data[tag] = None
    try:
        save_preset_file(path, data)
        print(f"Preset saved: {path}")
        refresh_presets_dropdown()
        dpg.set_value("preset_combo", os.path.splitext(os.path.basename(path))[0])
        return True
    except Exception as e:
        print("Failed to save preset:", e)
        return False


def save_preset_to_folder(name):
    if not name or not name.strip():
        print("Preset name empty.")
        return False
    try:
        path = preset_filename_from_name(name, PRESETS_DIR)
    except ValueError:
        print("Preset name invalid. Use alphanumerics, -, _, . and spaces.")
        return False

    if os.path.exists(path):
        try:
            preset_name = os.path.splitext(os.path.basename(path))[0]
            dpg.set_value("preset_to_overwrite_path", path)
            dpg.set_value("preset_to_overwrite_name", preset_name)
            dpg.set_value("overwrite_preset_name_display", f"Preset '{preset_name}' already exists. Overwrite it?")
            dpg.show_item("overwrite_preset_modal")
            return False
        except Exception:
            pass
    return _write_preset_file(path)


def confirm_overwrite_preset(sender=None, app_data=None):
    path = dpg.get_value("preset_to_overwrite_path")
    dpg.hide_item("overwrite_preset_modal")
    dpg.set_value("overwrite_preset_name_display", "")
    dpg.set_value("preset_to_overwrite_name", "")
    dpg.set_value("preset_to_overwrite_path", "")
    if not path:
        return
    _write_preset_file(path)


def cancel_overwrite_preset(sender=None, app_data=None):
    dpg.hide_item("overwrite_preset_modal")
    dpg.set_value("overwrite_preset_name_display", "")
    dpg.set_value("preset_to_overwrite_name", "")
    dpg.set_value("preset_to_overwrite_path", "")


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
        data = load_preset_file(path)
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
        delete_preset_file(path)
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
        dpg.configure_item("preset_combo", items=list_presets_on_disk(PRESETS_DIR))
        dpg.set_value("preset_to_delete", "")


def reset_to_defaults(sender=None, app_data=None):
    for tag, val in DEFAULT_PRESET.items():
        try:
            dpg.set_value(tag, val)
        except Exception:
            pass
    update_main_preview()
    print("Sliders reset to defaults.")


def load_image(sender=None, app_data=None):
    path = dpg.get_value("file")
    load_image_from_path(path)


def save_image(sender=None, app_data=None):
    if STATE.full_img is None:
        print("No image loaded.")
        return

    format_label = dpg.get_value("save_format") if dpg.does_item_exist("save_format") else SAVE_FORMAT_JPEG
    if format_label not in SAVE_FORMAT_SPECS:
        format_label = SAVE_FORMAT_JPEG

    input_path = dpg.get_value("file")
    if format_label == SAVE_FORMAT_TIFF and not is_tiff_path(input_path):
        print("TIFF export is only available when the loaded source file is TIFF.")
        update_save_format_options(input_path)
        return

    spec = SAVE_FORMAT_SPECS[format_label]
    if input_path:
        folder, name = os.path.split(input_path)
        base, _ = os.path.splitext(name)
        out_path = os.path.join(folder, base + "_IRG" + spec["ext"])
    else:
        out_path = os.path.join(os.getcwd(), "IRG_output_IRG" + spec["ext"])

    temp = dpg.get_value("wb_temp")
    tint = dpg.get_value("wb_tint")
    wb_full = apply_white_balance(STATE.full_img, temp, tint)
    out = scientific_irg_transform(wb_full)

    saved = save_output_array(out, out_path, format_label)
    if saved:
        try:
            if os.name == "posix":
                subprocess.run(["open", "-R", out_path])
            elif os.name == "nt":
                subprocess.run(["explorer", "/select,", out_path])
        except Exception:
            pass


def main():
    dpg.create_context()

    callbacks = {
        "open_file_callback": open_file_callback,
        "save_image": save_image,
        "save_preset_to_folder": save_preset_to_folder,
        "load_preset_from_folder": load_preset_from_folder,
        "show_delete_preset_confirm": show_delete_preset_confirm,
        "refresh_presets_dropdown": refresh_presets_dropdown,
        "confirm_delete_preset": confirm_delete_preset,
        "confirm_overwrite_preset": confirm_overwrite_preset,
        "cancel_overwrite_preset": cancel_overwrite_preset,
        "reset_to_defaults": reset_to_defaults,
        "update_main_preview": update_main_preview,
        "toggle_wb_dropper": toggle_wb_dropper,
        "on_marker_size_changed": on_marker_size_changed,
    }

    ui.build_ui(callbacks)

    dpg.create_viewport(title="Py-Chrome", width=1300, height=850)
    dpg.setup_dearpygui()

    # Add global mouse click handler for WB dropper
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=on_preview_image_click)

    # populate presets dropdown at start
    refresh_presets_dropdown()
    update_save_format_options(dpg.get_value("file") if dpg.does_item_exist("file") else None)

    # create initial scatter themes and bind them
    rebuild_scatter_themes(dpg.get_value("scatter_marker_size") if dpg.does_item_exist("scatter_marker_size") else DEFAULT_MARKER_SIZE)
    update_analysis_labels(dpg.get_value("scatter_use_converted") if dpg.does_item_exist("scatter_use_converted") else True)

    # initialize the histogram texture empty (correct shape: height x width x 4)
    empty_hist = np.zeros((HIST_H, HIST_W, 4), dtype=np.float32)
    dpg.set_value("hist_texture", _as_texture_value(empty_hist))

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()

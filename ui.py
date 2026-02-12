import dearpygui.dearpygui as dpg

from state import (
    CHANNEL_PREVIEWS,
    DEFAULT_MARKER_SIZE,
    DEFAULT_PRESET,
    HIST_H,
    HIST_W,
    MAX_CH_H,
    MAX_CH_W,
    MAX_PREVIEW_H,
    MAX_PREVIEW_W,
    SAVE_FORMAT_BMP,
    SAVE_FORMAT_JPEG,
    SAVE_FORMAT_PNG,
)


def build_ui(callbacks):
    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            width=MAX_PREVIEW_W,
            height=MAX_PREVIEW_H,
            default_value=[0.0] * (MAX_PREVIEW_W * MAX_PREVIEW_H * 4),
            tag="texture",
        )
        for ch in CHANNEL_PREVIEWS:
            dpg.add_dynamic_texture(
                width=MAX_CH_W,
                height=MAX_CH_H,
                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                tag=f"tex_{ch}",
            )
        for ch in CHANNEL_PREVIEWS:
            dpg.add_dynamic_texture(
                width=MAX_CH_W,
                height=MAX_CH_H,
                default_value=[0.0] * (MAX_CH_W * MAX_CH_H * 4),
                tag=f"tex_before_{ch}",
            )
        dpg.add_dynamic_texture(
            width=HIST_W,
            height=HIST_H,
            default_value=[0.0] * (HIST_W * HIST_H * 4),
            tag="hist_texture",
        )

    with dpg.window(label="Py-Chrome", width=1300, height=850):
        with dpg.group(horizontal=True):
            # Left panel (controls)
            with dpg.child_window(
                tag="left_panel",
                resizable_x=True,
                autosize_y=True,
                border=True,
                horizontal_scrollbar=True,
                width=500,
            ):
                dpg.add_input_text(label="Image Path", tag="file", width=400, readonly=True)

                # Top row: Open + Save IRG Image
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Open File...", callback=lambda: dpg.show_item("file_dialog"))
                    dpg.add_button(label="Save IRG Image", callback=callbacks["save_image"])
                    dpg.add_combo(
                        items=[SAVE_FORMAT_JPEG, SAVE_FORMAT_PNG, SAVE_FORMAT_BMP],
                        label="Save format",
                        tag="save_format",
                        width=130,
                    )
                dpg.add_text(
                    "TIFF export disabled for non-TIFF source files.",
                    tag="save_format_hint",
                    wrap=420,
                )

                dpg.add_spacer(height=6)

                # Collapsible Preset section
                with dpg.collapsing_header(label="Presets (folder)", default_open=True):
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(label="Preset name", tag="preset_name_input", width=90, default_value="")
                        dpg.add_button(
                            label="Save Preset to folder",
                            callback=lambda s, a: callbacks["save_preset_to_folder"](dpg.get_value("preset_name_input")),
                        )
                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        dpg.add_combo(
                            items=[],
                            label="Presets",
                            tag="preset_combo",
                            width=90,
                            callback=lambda s, a: callbacks["load_preset_from_folder"](dpg.get_value("preset_combo")),
                        )
                        dpg.add_button(label="Delete Preset", callback=callbacks["show_delete_preset_confirm"])
                        dpg.add_button(label="Refresh", callback=lambda s, a: callbacks["refresh_presets_dropdown"]())

                dpg.add_spacer(height=6)
                dpg.add_button(label="Reset to Default", callback=callbacks["reset_to_defaults"])
                dpg.add_spacer(height=6)

                # Histogram (outside scatterplots, below them)
                with dpg.collapsing_header(label="Histogram (IR→Red, Red→Green, Green→Blue)", default_open=True):
                    dpg.add_text(
                        "Histogram shows converted channels. Channel 0 (IR) is drawn in RED, channel 1 (Red) in GREEN, channel 2 (Green) in BLUE.",
                        wrap=420,
                        tag="hist_description",
                    )
                    # Visual gain slider to make small fractions visible (fixed-normalization maintained)
                    dpg.add_slider_float(
                        label="Histogram visual gain",
                        tag="hist_gain",
                        default_value=10.0,
                        min_value=1.0,
                        max_value=200.0,
                        callback=callbacks["update_main_preview"],
                    )
                    dpg.add_image("hist_texture", width=HIST_W, height=HIST_H)

                with dpg.collapsing_header(label="White Balance (Temperature + Tint)", default_open=True):
                    dpg.add_button(label="Set WB Reference", tag="wb_dropper_btn", callback=callbacks["toggle_wb_dropper"])
                    dpg.add_text(
                        "Click button, then click a gray area in the image to auto-adjust WB.",
                        wrap=400,
                    )
                    dpg.add_spacer(height=4)
                    dpg.add_slider_int(
                        label="WB Temperature (K)",
                        tag="wb_temp",
                        default_value=DEFAULT_PRESET["wb_temp"],
                        min_value=2000,
                        max_value=12000,
                        callback=callbacks["update_main_preview"],
                    )
                    dpg.add_slider_int(
                        label="WB Tint",
                        tag="wb_tint",
                        default_value=DEFAULT_PRESET["wb_tint"],
                        min_value=-100,
                        max_value=100,
                        callback=callbacks["update_main_preview"],
                    )
                    dpg.add_text(
                        "Temperature maps to color temperature (Kelvin). Tint is green<->magenta axis.",
                        wrap=400,
                    )

                dpg.add_spacer(height=6)

                with dpg.collapsing_header(label="Fraction sliders", default_open=True):
                    for label, tag, default, minv, maxv in [
                        ("Red Vis Fraction", "fracRx", DEFAULT_PRESET["fracRx"], 0.0, 1.0),
                        ("Green Vis Fraction", "fracGx", DEFAULT_PRESET["fracGx"], 0.0, 1.0),
                        ("IR Fraction", "fracBY", DEFAULT_PRESET["fracBY"], 0.0, 1.0),
                    ]:
                        dpg.add_slider_float(
                            label=label,
                            tag=tag,
                            default_value=default,
                            min_value=minv,
                            max_value=maxv,
                            callback=callbacks["update_main_preview"],
                        )

                dpg.add_spacer(height=6)

                with dpg.collapsing_header(label="Gamma & Exposure", default_open=True):
                    for label, tag, default, minv, maxv in [
                        ("Gamma Red Visible", "gammaRx", DEFAULT_PRESET["gammaRx"], 0.1, 5.0),
                        ("Gamma Red IR", "gammaRy", DEFAULT_PRESET["gammaRy"], 0.1, 5.0),
                        ("Gamma Green Visible", "gammaGx", DEFAULT_PRESET["gammaGx"], 0.1, 5.0),
                        ("Gamma Green IR", "gammaGy", DEFAULT_PRESET["gammaGy"], 0.1, 5.0),
                        ("Gamma IR", "gammaBY", DEFAULT_PRESET["gammaBY"], 0.1, 5.0),
                        ("Exposure", "exposure", DEFAULT_PRESET["exposure"], 0.1, 5.0),
                    ]:
                        dpg.add_slider_float(
                            label=label,
                            tag=tag,
                            default_value=default,
                            min_value=minv,
                            max_value=maxv,
                            callback=callbacks["update_main_preview"],
                        )

                dpg.add_text("", tag="warning_text", color=(255, 0, 0))
                dpg.add_spacer(height=6)

                # Scatterplots
                with dpg.collapsing_header(label="Scatterplots (After/Before Consistent Order)", default_open=True):
                    dpg.add_slider_int(
                        label="Scatter marker size",
                        tag="scatter_marker_size",
                        default_value=DEFAULT_MARKER_SIZE,
                        min_value=1,
                        max_value=8,
                        callback=callbacks["on_marker_size_changed"],
                    )
                    dpg.add_spacer(height=6)
                    dpg.add_checkbox(
                        label="Use Converted Data",
                        tag="scatter_use_converted",
                        default_value=True,
                        callback=callbacks["update_main_preview"],
                    )
                    with dpg.group(horizontal=False):
                        with dpg.plot(label="Red vs IR", height=170, width=-1, tag="plot_rg"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Red (0-255)", tag="axis_rg_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="IR (0-255)", tag="axis_rg_y")
                            dpg.set_axis_limits("axis_rg_x", 0.0, 255.0)
                            dpg.set_axis_limits("axis_rg_y", 0.0, 255.0)
                            dpg.add_scatter_series([], [], parent="axis_rg_y", tag="series_rg")
                        with dpg.plot(label="Green vs IR", height=170, width=-1, tag="plot_rb"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Green (0-255)", tag="axis_rb_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="IR (0-255)", tag="axis_rb_y")
                            dpg.set_axis_limits("axis_rb_x", 0.0, 255.0)
                            dpg.set_axis_limits("axis_rb_y", 0.0, 255.0)
                            dpg.add_scatter_series([], [], parent="axis_rb_y", tag="series_rb")
                        with dpg.plot(label="Red vs Green", height=170, width=-1, tag="plot_gb"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Red (0-255)", tag="axis_gb_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="Green (0-255)", tag="axis_gb_y")
                            dpg.set_axis_limits("axis_gb_x", 0.0, 255.0)
                            dpg.set_axis_limits("axis_gb_y", 0.0, 255.0)
                            dpg.add_scatter_series([], [], parent="axis_gb_y", tag="series_gb")

            # file dialog
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=callbacks["open_file_callback"],
                tag="file_dialog",
                width=700,
                height=400,
            ):
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
    with dpg.window(
        label="Confirm delete preset",
        modal=True,
        show=False,
        tag="delete_preset_modal",
        no_title_bar=False,
        width=400,
        height=120,
    ):
        dpg.add_text("Are you sure you want to permanently delete this preset?")
        dpg.add_spacer(height=6)
        dpg.add_text("", tag="delete_preset_name_display")
        dpg.add_spacer(height=6)
        dpg.add_text("", tag="preset_to_delete", show=False)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Confirm Delete", callback=callbacks["confirm_delete_preset"])
            dpg.add_button(label="Cancel", callback=lambda s, a: dpg.hide_item("delete_preset_modal"))

    # Preset overwrite confirmation modal
    with dpg.window(
        label="Preset exists",
        modal=True,
        show=False,
        tag="overwrite_preset_modal",
        no_title_bar=False,
        width=420,
        height=140,
    ):
        dpg.add_text("", tag="overwrite_preset_name_display", wrap=390)
        dpg.add_text("", tag="preset_to_overwrite_name", show=False)
        dpg.add_text("", tag="preset_to_overwrite_path", show=False)
        dpg.add_spacer(height=8)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Overwrite", callback=callbacks["confirm_overwrite_preset"])
            dpg.add_button(label="Cancel", callback=callbacks["cancel_overwrite_preset"])

Py-Chrome V9
============

Overview
--------
Py-Chrome V9 is a desktop tool for IR-style photo conversion and analysis made to emulate CIR, formally known as Aerochrome.
It provides a live preview pipeline, white-balance tools, scatter/histogram diagnostics,
preset management, and export options from a DearPyGui interface.

This project is now modular:
- app.py: application entrypoint and callbacks
- ui.py: DearPyGui layout and widget creation
- processing.py: image math, rendering helpers, histogram/scatter processing
- io_ops.py: image load/save logic and format handling
- presets.py: preset file operations
- state.py: shared constants and app state
- Py-Chrome V9_8.py: compatibility launcher that calls app.main()

Fast Install (MacOS)
-------------------

1. Install Python (https://www.python.org/downloads/)
2. Download all contents and place all .py files together in a folder
3. Right-click `installer.py` and open with Python Launcher
4. The finished application package is created in:
   - `release/PyChromeV9`
   - plus a ZIP in `release/` (unless `--no-zip` is used)
5. Optional: Delete all source code and use the exported application

Run From Source
-------------------

1. Upgrade pip:
   python3 -m pip install --upgrade pip

2. Install required packages:
   pip install numpy pillow dearpygui tifffile

3. Run the app:
   app.py


Required Python Packages
------------------------
Required:
- numpy
  Why: core array math for per-pixel processing, histogram/scatter data prep.
- pillow
  Why: image loading/resizing and non-TIFF export support.
- dearpygui
  Why: desktop UI toolkit used for all controls, plots, and image textures.

Strongly recommended:
- tifffile
  Why: enables high-precision TIFF load/save and 16-bit TIFF export path.

Optional (build only):
- pyinstaller
  Why: creates standalone executable distributions.


Run and Build
-------------
Run from source:
- python3 app.py

Build standalone executable (example):
- python3 -m PyInstaller --noconfirm --clean --onefile --windowed --name "PyChromeV9" --collect-all dearpygui --collect-all PIL app.py

Current build outputs (if already built):
- dist/PyChromeV9
- dist/PyChromeV9.app

Note on presets path:
- Source run: presets are stored in ./presets relative to the project.
- Frozen executable: presets are stored next to the executable in a presets folder.


Features: What They Do and Why
------------------------------
1. Open image file
   What: loads TIFF/JPG/PNG sources and prepares full-resolution + preview arrays.
   Why: gives fast interactive edits while preserving full-res data for export.

2. Save with selectable output format
   What: export as TIFF, JPEG, PNG, or BMP (TIFF shown only when TIFF source is loaded).
   Why: supports common delivery formats while protecting TIFF-specific workflows.

3. White Balance dropper + sliders
   What: click a neutral area to estimate WB temperature/tint, or adjust manually.
   Why: IR conversions are highly WB-sensitive; this speeds accurate color setup.

4. Fraction controls (Red/Green visible fractions and IR fraction)
   What: tunes channel contribution to the scientific IRG transform.
   Why: gives control over channel separation and highlight behavior.

5. Gamma and exposure controls
   What: adjust per-channel gamma response and global exposure.
   Why: provides tonal shaping and dynamic-range balancing in output.

6. Scientific IRG transform
   What: converts preview/full image using the IRG model equation.
   Why: core algorithm for the Aerochrome-style conversion workflow.

7. Main preview + before/after channel tiles
   What: shows transformed preview plus channel-level mini-previews.
   Why: makes it easier to debug channel behavior and tune settings quickly.

8. Histogram with visual gain
   What: displays per-channel frequency with adjustable visual gain.
   Why: helps detect clipping/distribution issues while keeping normalization stable.

9. Scatter plots (before/after mode)
   What: plots channel relationships in fixed order with optional converted/source mode.
   Why: gives quick diagnostics for channel separation and transform response.

10. Presets (save/load/delete/overwrite confirmation)
    What: stores slider states to JSON in presets folder; confirms overwrite.
    Why: enables repeatable looks and safer preset management.

11. Reset to default
    What: restores all tuned parameters to known defaults.
    Why: allows quick recovery from extreme settings during experimentation.

12. Performance optimizations
    What: numpy texture uploads, cached scatter sampling, vectorized histogram fill,
          single-pass channel preview resize, label-update guard.
    Why: keeps UI interaction responsive during frequent slider updates.


Troubleshooting
---------------
Issue: TIFF export not available.
- Cause:
  Source file is not TIFF, or save option not applicable.
- Action:
  Load a TIFF source file if you need TIFF output.

Issue: TIFF export works but appears 8-bit.
- Cause:
  `tifffile` missing or failed, fallback path used.
- Action:
  Reinstall `tifffile` in active environment.

Issue: Image loads but colors look wrong immediately.
- Cause:
  WB not neutralized before transform tuning.
- Action:
  Use WB dropper first, then retune fractions/gamma.

Issue: Scatter seems noisy.
- Cause:
  Large scene variance + capped random sample.
- Action:
  Compare with a cleaner reference image and keep consistent preset baseline.

Issue: Preset missing from dropdown.
- Cause:
  file removed/renamed or invalid path state.
- Action:
  Press Refresh; verify file exists in `presets` folder.


License / Use
-------------
MIT

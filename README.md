Py-Chrome V9
============
![py-chrome-version-9-available-v0-re1k9c2qf1jg1](https://github.com/user-attachments/assets/302d0313-5478-4b25-bed2-9d1d6c8c9727)

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
   Why: IR conversions are highly WB-sensitive; this speeds accurate color setup. The WB affects the image BEFORE conversion

4. Fraction controls (Red/Green visible fractions and IR fraction)
   What: tunes channel contribution to the scientific IRG transform.
   Why: gives control over channel separation and highlight behavior. The smaller the fraction of visible light, the greater the subtraction and saturation

   Example of no subtraction
   <img width="1461" height="1071" alt="Screenshot 2026-02-13 at 8 34 18 PM" src="https://github.com/user-attachments/assets/2e8d8c90-3874-4235-9338-4b0d2a808e8f" />

   Example of high subtraction
   <img width="1461" height="1071" alt="Screenshot 2026-02-13 at 8 35 28 PM" src="https://github.com/user-attachments/assets/04ff77fa-2f0a-42db-9f81-caa3c4a25b78" />

   Note: Low Visible Channel fraction can cause artifacting in highlights

6. Gamma and exposure controls
   What: adjust per-channel gamma response and global exposure. Higher Gamma = Decrease in brightness &           Vice Versa
   Why: provides tonal shaping and dynamic-range balancing in output.

      - Gamma Red/Green Visible
         What it does: Affects overall brightness of that specific channel globally
         
      - Gamma Red/Green IR
         What it does: Affects brightness of channel in areas where subtraction takes place (e.g foilage)
         
   Example 1: To get crimson red foilage, increase Gamma Red and Green IR to decrease brightness in foilage area. Adjust Gamma Visible slightly if needed to re-balance overall color
         <img width="1461" height="1041" alt="Screenshot 2026-02-13 at 8 13 08 PM" src="https://github.com/user-attachments/assets/bde572f1-0cb3-497a-8465-0e3c87c2979e" />

   Example 2: To extract color variation from foilage, increase Gamma in Red and Green IR while decreasing Gamma in the Red and Green Visible
<img width="1461" height="1041" alt="Screenshot 2026-02-13 at 8 29 24 PM" src="https://github.com/user-attachments/assets/5111ee76-6351-4971-b374-5de4c868dd7b" />


      - Gamma IR
         What it does: Affects IR channel brightness; Red and Green channel compensated automatically
         Example: Increasing Gamma IR decreases IR channel brightness while increasing both Red and                        Green channel equally - useful for minor refinement

7. Scientific IRG transform
   What: converts preview/full image using the IRG model equation.
   Why: core algorithm for the Aerochrome-style conversion workflow.

   Simplified Equation Example:
   
   [R] = [R + IR] - [IR]
   
   [G] = [G + IR] - [IR]

   Note: Because we do not exactly know how much IR contaminates the Red and Green channel, it is best to treat this as visual aesthetic per scene rather than attempting to be "Correct"

9. Main preview + before/after channel tiles
   What: shows transformed preview plus channel-level mini-previews.
   Why: makes it easier to debug channel behavior and tune settings quickly.

10. Histogram with visual gain
   What: displays per-channel frequency with adjustable visual gain.
   Why: helps detect clipping/distribution issues while keeping normalization stable.

11. Scatter plots (before/after mode)
   What: plots channel relationships in fixed order with optional converted/source mode.
   Why: gives quick diagnostics for channel separation and transform response.

12. Presets (save/load/delete/overwrite confirmation)
    What: stores slider states to JSON in presets folder; confirms overwrite.
    Why: enables repeatable looks and safer preset management.

13. Reset to default
    What: restores all tuned parameters to known defaults.
    Why: allows quick recovery from extreme settings during experimentation.

14. Performance optimizations
    What: numpy texture uploads, cached scatter sampling, vectorized histogram fill,
          single-pass channel preview resize, label-update guard.
    Why: keeps UI interaction responsive during frequent slider updates.
    
Typical Workflow (Recommended)
------------------------------
Use this sequence for effective and consistent results:
1. Open one representative image from your session.
2. Set WB using "Set WB Reference" and click a neutral area.
3. Fine-tune WB sliders manually if needed.
4. Tune fraction sliders (`fracRx`, `fracGx`, `fracBY`) first.
5. Tune gamma sliders second for tonal shaping.
6. Tune exposure last.
7. Watch histogram for clipping patterns.
8. Check scatter plots for channel separation behavior.
9. Save preset with a session-specific name.
10. (If starting a new session again after close) Load next image and apply same preset.
11. Adjust only WB/exposure per image if capture conditions drifted.
12. Export to final format.

Troubleshooting
---------------
Issue: The Open File box does not appear
- Cause:
  The Open File is behind the main interface
- Action: Top left of the window, next to Py-Chrome text, click on window to minimize main window to see open file box or other dialogue

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

Photo Examples
-------------
Additional editing done in Lightroom

![r05jqjyab7hg1](https://github.com/user-attachments/assets/033a1ad5-1cca-4c9c-b24a-8b6b7b55f7e7)
![after-before-reference-faithful-recreation-v0-2gkqm8nyb7hg1](https://github.com/user-attachments/assets/bda34fc6-02a1-49d7-8807-68ff58c124ef)
![winding-road-aerochrome-emulation-v0-rubf56zrlkhg1](https://github.com/user-attachments/assets/adb76a3f-d87b-4ce9-9f93-a72e283570b5)
![cir-symmetry-v0-mh69rd4e8tig1](https://github.com/user-attachments/assets/441dd903-545f-4458-a600-7d010f852351)


License / Use
-------------
MIT

import json
import os


def preset_filename_from_name(name, presets_dir):
    safe = "".join(c for c in name if c.isalnum() or c in "-_. ").strip()
    if not safe:
        raise ValueError("Invalid preset name")
    return os.path.join(presets_dir, safe + ".json")


def list_presets_on_disk(presets_dir):
    items = []
    try:
        for fname in sorted(os.listdir(presets_dir)):
            if fname.lower().endswith(".json"):
                items.append(os.path.splitext(fname)[0])
    except Exception:
        pass
    return items


def save_preset_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_preset_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_preset_file(path):
    os.remove(path)

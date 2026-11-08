"""Small shared helpers for finding images and their YOLO label files."""
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(src):
    """Yield every image under `src` (recursive, sorted)."""
    for p in sorted(Path(src).rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            yield p


def label_path_for(img):
    """Return the YOLO label path for an image.

    Handles the common layouts:
      folder/images/x.jpg  -> folder/labels/x.txt   (YOLO / Roboflow export)
      folder/x.jpg         -> folder/x.txt          (labelImg, same folder)
    """
    img = Path(img)
    if img.parent.name == "images":
        return img.parent.parent / "labels" / (img.stem + ".txt")
    return img.with_suffix(".txt")


_ALIASES = {
    "motorcycle": {"motorcycle", "motorbike", "motor", "moto"},
    "rider": {"rider", "person", "driver"},
    "helmet": {"helmet", "with_helmet", "with helmet", "helmeted"},
}


def class_ids(names):
    """Find the ids of motorcycle, rider, helmet in a model/dataset `names` mapping.

    `names` is a dict {id: name} or a list. Matching is by name, so the class
    ORDER in your Roboflow dataset does not matter. Returns (MOTO, RIDER, HELMET).
    """
    items = names.items() if isinstance(names, dict) else enumerate(names)
    by_name = {str(n).strip().lower(): int(i) for i, n in items}
    found = {}
    for key, aliases in _ALIASES.items():
        hit = [by_name[a] for a in aliases if a in by_name]
        if not hit:
            raise SystemExit(
                f"No class for '{key}' in {by_name}. Rename your classes in Roboflow "
                f"to motorcycle / rider / helmet (or add your name to _ALIASES in utils.py)."
            )
        found[key] = hit[0]
    return found["motorcycle"], found["rider"], found["helmet"]

from pathlib import Path

import cv2
import numpy as np
import yaml

# ====== KONFIG ======
DATA_YAML = r"C:\Users\dyksi\OneDrive\Pulpit\magisterka\yolo_test\trening\dataset_v8_linie_500\data.yaml"
SHIFT_LABELS = True   # 0=tło, klasy będą 1..nc

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# ====== POMOCNICZE ======
def load_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names_field):
    if isinstance(names_field, dict):
        return [names_field[k] for k in sorted(names_field, key=lambda x: int(x))]
    if isinstance(names_field, list):
        return names_field
    return []


def class_to_pixel(cls_id: int) -> int:
    return cls_id + 1 if SHIFT_LABELS else cls_id


def resolve_split_path(data_yaml_path: Path, data: dict, split_key: str) -> Path:
    base_dir = data_yaml_path.parent.resolve()
    raw = str(data[split_key]).replace("\\", "/").strip()
    p = Path(raw)

    candidates = []

    if p.is_absolute():
        candidates.append(p.resolve())

    candidates.append((base_dir / p).resolve())

    yaml_root = data.get("path", None)
    if yaml_root:
        candidates.append((base_dir / str(yaml_root) / p).resolve())

    raw_trim = raw
    while raw_trim.startswith("../"):
        raw_trim = raw_trim[3:]
        candidates.append((base_dir / raw_trim).resolve())

    parts = Path(raw).parts
    if len(parts) >= 2:
        tail2 = Path(*parts[-2:])
        candidates.append((base_dir / tail2).resolve())

    uniq = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            uniq.append(c)
            seen.add(s)

    for c in uniq:
        if c.exists():
            return c

    print(f"[DEBUG] Nie znaleziono istniejącej ścieżki dla '{split_key}'. Próbowane warianty:")
    for c in uniq:
        print("   ", c)

    return uniq[0]


def polygon_line_to_points(parts, w: int, h: int):
    if len(parts) < 7:
        return None, "Za mało punktów dla polygonu"

    try:
        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
    except ValueError:
        return None, "Nie da się sparsować liczb"

    if len(coords) == 4:
        return None, "To wygląda jak YOLO detection bbox, nie segmentacja"

    if len(coords) % 2 != 0:
        return None, "Nieparzysta liczba współrzędnych"

    pts = []
    for i in range(0, len(coords), 2):
        x = int(round(coords[i] * w))
        y = int(round(coords[i + 1] * h))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        pts.append([x, y])

    if len(pts) < 3:
        return None, "Polygon ma mniej niż 3 punkty"

    return (cls_id, np.array([pts], dtype=np.int32)), None


def yolo_seg_to_mask(image_path: Path, label_path: Path, num_classes: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Nie mogę wczytać obrazu: {image_path}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not label_path.exists():
        return mask

    bad_lines = []

    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            result, err = polygon_line_to_points(parts, w, h)

            if err is not None:
                bad_lines.append(f"{label_path.name}:{line_no} -> {err}")
                continue

            cls_id, pts = result

            if cls_id < 0 or cls_id >= num_classes:
                bad_lines.append(
                    f"{label_path.name}:{line_no} -> cls_id={cls_id} poza zakresem 0..{num_classes - 1}"
                )
                continue

            cv2.fillPoly(mask, pts, color=class_to_pixel(cls_id))

    if bad_lines:
        print(f"[WARN] Problemy w pliku: {label_path}")
        for msg in bad_lines[:10]:
            print("   ", msg)
        if len(bad_lines) > 10:
            print(f"    ... i jeszcze {len(bad_lines) - 10} problematycznych linii")

    return mask


def process_split(images_dir: Path, num_classes: int):
    split_dir = images_dir.parent
    labels_dir = split_dir / "labels"
    masks_dir = split_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        print(f"[WARN] Brak katalogu labels: {labels_dir}")
        return

    image_paths = []
    for ext in IMG_EXTS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"[WARN] Brak obrazów w {images_dir}")
        return

    saved = 0
    empty_labels = 0

    for img_path in image_paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            empty_labels += 1

        mask = yolo_seg_to_mask(img_path, lbl_path, num_classes)
        out_path = masks_dir / f"{img_path.stem}.png"

        ok = cv2.imwrite(str(out_path), mask)
        if not ok:
            print(f"[WARN] Nie udało się zapisać maski: {out_path}")
            continue

        saved += 1

    print(f"[OK] {split_dir.name}: zapisano {saved} masek do {masks_dir}")
    if empty_labels > 0:
        print(f"[INFO] {split_dir.name}: {empty_labels} obrazów bez labeli -> zapisane jako samo tło")


def main():
    data_yaml_path = Path(DATA_YAML).resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Nie istnieje data.yaml: {data_yaml_path}")

    data = load_data_yaml(data_yaml_path)

    names = normalize_names(data.get("names", []))
    nc = int(data.get("nc", len(names)))

    print(f"[INFO] data.yaml: {data_yaml_path}")
    print(f"[INFO] Liczba klas: {nc}")
    print(f"[INFO] Klasy: {names}")

    found_any_split = False

    for key in ("train", "val", "valid", "test"):
        if key not in data or data[key] is None:
            continue

        images_dir = resolve_split_path(data_yaml_path, data, key)

        if not images_dir.exists():
            print(f"[WARN] Nie istnieje ścieżka dla '{key}': {images_dir}")
            continue

        print(f"[INFO] Przetwarzam split '{key}': {images_dir}")
        process_split(images_dir, num_classes=nc)
        found_any_split = True

    if not found_any_split:
        print("[WARN] Nie znaleziono żadnego poprawnego splitu do przetworzenia.")

    print("\nMapowanie wartości pikseli w maskach:")
    if SHIFT_LABELS:
        print("0 = tło")
        for i, name in enumerate(names):
            print(f"{i + 1} = {name}")
        print(f"\nDo DeepLabV3 ustaw: NUM_CLASSES = {nc + 1}")
    else:
        for i, name in enumerate(names):
            print(f"{i} = {name}")
        print(f"\nDo DeepLabV3 ustaw: NUM_CLASSES = {nc}")

    print("\nGotowe.")


if __name__ == "__main__":
    main()

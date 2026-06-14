from pathlib import Path
import csv
import shutil
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ====== KONFIG ======
DATASET_ROOT = Path(r"C:\Users\dyksi\OneDrive\Pulpit\magisterka\yolo_test\trening\dataset_v8_linie_500")

# Dla SHIFT_LABELS=True w mapperze:
# 0 = background, 1..N = klasy właściwe
NUM_CLASSES = 4
CLASS_NAMES = ["background", "red_line", "white_line", "yellow_line"]

IMG_SIZE = 512
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 50
NUM_WORKERS = 4
SAVE_BEST_BY = "iou"   # "iou" albo "dice"

MODEL_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
DO_REDUCE_LABELS = False  # dla masek 0=background, 1..N=klasy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
AMP_ENABLED = DEVICE.type == "cuda"

RUNS_ROOT = Path(__file__).resolve().parent / "runs_segformer"
RUN_NAME = "train"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# ====== POMOCNICZE ======
def list_images(images_dir: Path):
    images = []
    for ext in IMG_EXTS:
        images.extend(images_dir.glob(f"*{ext}"))
    return sorted(images)


def make_run_dir(root: Path, name: str = "train") -> Path:
    root.mkdir(parents=True, exist_ok=True)

    run_dir = root / name
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    i = 2
    while True:
        run_dir = root / f"{name}{i}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        i += 1


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def save_config(run_dir: Path):
    cfg = []
    cfg.append(f"DATASET_ROOT={DATASET_ROOT}")
    cfg.append(f"NUM_CLASSES={NUM_CLASSES}")
    cfg.append(f"CLASS_NAMES={CLASS_NAMES}")
    cfg.append(f"IMG_SIZE={IMG_SIZE}")
    cfg.append(f"BATCH_SIZE={BATCH_SIZE}")
    cfg.append(f"LR={LR}")
    cfg.append(f"EPOCHS={EPOCHS}")
    cfg.append(f"NUM_WORKERS={NUM_WORKERS}")
    cfg.append(f"SAVE_BEST_BY={SAVE_BEST_BY}")
    cfg.append(f"MODEL_CHECKPOINT={MODEL_CHECKPOINT}")
    cfg.append(f"DO_REDUCE_LABELS={DO_REDUCE_LABELS}")
    cfg.append(f"DEVICE={DEVICE}")
    cfg.append(f"AMP_ENABLED={AMP_ENABLED}")
    cfg.append(f"RUNS_ROOT={RUNS_ROOT}")
    cfg.append(f"RUN_NAME={RUN_NAME}")
    save_text(run_dir / "train_config.txt", "\n".join(cfg))


def save_history_csv(history, csv_path: Path):
    fieldnames = [
        "epoch",
        "train_loss", "train_iou", "train_dice",
        "val_loss", "val_iou", "val_dice",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_curve(history, train_key, val_key, ylabel, out_path: Path):
    epochs = [x["epoch"] for x in history]
    train_vals = [x[train_key] for x in history]
    val_vals = [x[val_key] for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_vals, label=f"train {ylabel.lower()}")
    plt.plot(epochs, val_vals, label=f"val {ylabel.lower()}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def update_confusion_matrix(cm: np.ndarray, pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)

    valid = (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]

    idx = num_classes * target + pred
    binc = torch.bincount(idx, minlength=num_classes ** 2)
    cm += binc.reshape(num_classes, num_classes).cpu().numpy()


def metrics_from_confusion_matrix(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    iou_denom = tp + fp + fn
    dice_denom = 2 * tp + fp + fn

    iou_per_class = np.full_like(tp, np.nan, dtype=np.float64)
    dice_per_class = np.full_like(tp, np.nan, dtype=np.float64)

    valid_iou = iou_denom > 0
    valid_dice = dice_denom > 0

    iou_per_class[valid_iou] = tp[valid_iou] / iou_denom[valid_iou]
    dice_per_class[valid_dice] = (2 * tp[valid_dice]) / dice_denom[valid_dice]

    mean_iou = float(np.nanmean(iou_per_class)) if np.any(valid_iou) else 0.0
    mean_dice = float(np.nanmean(dice_per_class)) if np.any(valid_dice) else 0.0

    return mean_iou, mean_dice, iou_per_class, dice_per_class


def save_confusion_matrix_image(cm: np.ndarray, class_names, out_path: Path, normalize: bool = False):
    mat = cm.astype(np.float64).copy()
    title = "Confusion Matrix"

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums
        title += " (Normalized)"
    else:
        title += " (Raw)"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = f"{mat[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_per_class_metrics_csv(class_names, iou_per_class, dice_per_class, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "iou", "dice"])
        for i, name in enumerate(class_names):
            iou_val = "" if np.isnan(iou_per_class[i]) else float(iou_per_class[i])
            dice_val = "" if np.isnan(dice_per_class[i]) else float(dice_per_class[i])
            writer.writerow([i, name, iou_val, dice_val])


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    palette = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [255, 255, 255],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128],
    ], dtype=np.uint8)

    if num_classes > len(palette):
        raise ValueError("Za mała paleta kolorów dla liczby klas.")

    return palette[mask]


def build_id2label(class_names):
    return {i: name for i, name in enumerate(class_names)}


# ====== DATASET ======
class SegFormerDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, processor: SegformerImageProcessor, num_classes: int):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        self.num_classes = int(num_classes)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Nie istnieje katalog images: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Nie istnieje katalog masks: {self.masks_dir}")

        self.images = list_images(self.images_dir)
        if len(self.images) == 0:
            raise RuntimeError(f"Brak obrazów w: {self.images_dir}")

        missing_masks = []
        for img_path in self.images:
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                missing_masks.append(mask_path)
        if missing_masks:
            raise RuntimeError(
                f"Brakuje {len(missing_masks)} masek. Przykład: {missing_masks[0]}"
            )

    def __len__(self):
        return len(self.images)

    def load_image_and_mask(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Nie mogę wczytać obrazu: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Nie mogę wczytać maski: {mask_path}")

        uniq = np.unique(mask)
        if uniq.max(initial=0) >= self.num_classes:
            raise ValueError(
                f"Maska {mask_path} ma wartości {uniq.tolist()}, a NUM_CLASSES={self.num_classes}."
            )

        return img_path, img_rgb, mask.astype(np.uint8)

    def __getitem__(self, idx):
        _, image, mask = self.load_image_and_mask(idx)

        enc = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        pixel_values = enc["pixel_values"].squeeze(0)
        labels = enc["labels"].squeeze(0).to(torch.long)
        return pixel_values, labels


# ====== MODEL ======
def build_model(num_classes: int, class_names, checkpoint: str):
    id2label = build_id2label(class_names)
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def upsample_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits


def save_prediction_examples(model, dataset, processor, device, out_dir: Path, num_examples: int = 6):
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    n = min(num_examples, len(dataset))

    with torch.no_grad():
        for idx in range(n):
            _, image_rgb, gt_mask = dataset.load_image_and_mask(idx)

            enc = processor(images=image_rgb, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device)

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=gt_mask.shape,
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            if image_rgb.shape[:2] != gt_mask.shape:
                image_rgb = cv2.resize(image_rgb, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

            gt_vis = colorize_mask(gt_mask, NUM_CLASSES)
            pred_vis = colorize_mask(pred, NUM_CLASSES)
            canvas = np.concatenate([image_rgb, gt_vis, pred_vis], axis=1)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(out_dir / f"sample_{idx:02d}.png"), canvas)


# ====== TRAIN / VAL ======
def run_epoch(model, loader, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for pixel_values, labels in tqdm(loader, leave=False):
        pixel_values = pixel_values.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * pixel_values.size(0)
        n += pixel_values.size(0)

        logits_up = upsample_logits(logits, labels)
        preds = torch.argmax(logits_up, dim=1)
        update_confusion_matrix(cm, preds, labels, NUM_CLASSES)

    epoch_loss = total_loss / max(n, 1)
    mean_iou, mean_dice, iou_per_class, dice_per_class = metrics_from_confusion_matrix(cm)

    return {
        "loss": epoch_loss,
        "iou": mean_iou,
        "dice": mean_dice,
        "cm": cm,
        "iou_per_class": iou_per_class,
        "dice_per_class": dice_per_class,
    }


def main():
    if len(CLASS_NAMES) != NUM_CLASSES:
        raise ValueError("CLASS_NAMES musi mieć tyle elementów co NUM_CLASSES.")

    run_dir = make_run_dir(RUNS_ROOT, RUN_NAME)
    print(f"[INFO] Run dir: {run_dir}")
    save_config(run_dir)

    train_images = DATASET_ROOT / "train" / "images"
    train_masks = DATASET_ROOT / "train" / "masks"
    val_images = DATASET_ROOT / "val" / "images"
    val_masks = DATASET_ROOT / "val" / "masks"

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_CHECKPOINT,
        do_reduce_labels=DO_REDUCE_LABELS,
        size={"height": IMG_SIZE, "width": IMG_SIZE},
    )

    train_ds = SegFormerDataset(train_images, train_masks, processor=processor, num_classes=NUM_CLASSES)
    val_ds = SegFormerDataset(val_images, val_masks, processor=processor, num_classes=NUM_CLASSES)

    print(f"[INFO] Train: {len(train_ds)} obrazów")
    print(f"[INFO] Val:   {len(val_ds)} obrazów")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Checkpoint: {MODEL_CHECKPOINT}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = build_model(NUM_CLASSES, CLASS_NAMES, MODEL_CHECKPOINT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    history = []
    best_score = -1.0
    best_epoch = -1
    best_cm = None
    best_iou_per_class = None
    best_dice_per_class = None

    best_dir = run_dir / "best_model"
    last_dir = run_dir / "last_model"
    results_csv_path = run_dir / "results.csv"

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, scaler=scaler)
        val_metrics = run_epoch(model, val_loader, optimizer=None, scaler=None)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
        }
        history.append(row)
        save_history_csv(history, results_csv_path)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {row['train_loss']:.4f} IoU {row['train_iou']:.4f} Dice {row['train_dice']:.4f} || "
            f"val loss {row['val_loss']:.4f} IoU {row['val_iou']:.4f} Dice {row['val_dice']:.4f}"
        )

        if last_dir.exists():
            shutil.rmtree(last_dir)
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        score = row["val_iou"] if SAVE_BEST_BY == "iou" else row["val_dice"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_cm = val_metrics["cm"].copy()
            best_iou_per_class = val_metrics["iou_per_class"].copy()
            best_dice_per_class = val_metrics["dice_per_class"].copy()

            if best_dir.exists():
                shutil.rmtree(best_dir)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  [SAVE] best_model/ (best val {SAVE_BEST_BY} = {best_score:.4f})")

            examples_dir = run_dir / "val_examples"
            if examples_dir.exists():
                shutil.rmtree(examples_dir)
            save_prediction_examples(model, val_ds, processor, DEVICE, examples_dir, num_examples=6)

        plot_curve(history, "train_loss", "val_loss", "Loss", run_dir / "loss.png")
        plot_curve(history, "train_iou", "val_iou", "IoU", run_dir / "iou.png")
        plot_curve(history, "train_dice", "val_dice", "Dice", run_dir / "dice.png")

    total_training_time = time.time() - start_time

    print(f"\n[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best val {SAVE_BEST_BY}: {best_score:.4f}")
    print(f"[INFO] Łączny czas treningu: {total_training_time:.2f} s")

    if best_cm is not None:
        save_confusion_matrix_image(best_cm, CLASS_NAMES, run_dir / "confusion_matrix_raw.png", normalize=False)
        save_confusion_matrix_image(best_cm, CLASS_NAMES, run_dir / "confusion_matrix_norm.png", normalize=True)
        save_per_class_metrics_csv(CLASS_NAMES, best_iou_per_class, best_dice_per_class, run_dir / "per_class_metrics.csv")

    print(f"[INFO] Zapisano do: {run_dir}")
    print("[INFO] Pliki:")
    print("       - best_model/")
    print("       - last_model/")
    print("       - results.csv")
    print("       - loss.png")
    print("       - iou.png")
    print("       - dice.png")
    print("       - confusion_matrix_raw.png")
    print("       - confusion_matrix_norm.png")
    print("       - per_class_metrics.csv")
    print("       - train_config.txt")
    print("       - val_examples/")


if __name__ == "__main__":
    main()
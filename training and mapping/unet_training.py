from pathlib import Path
import csv
import shutil
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A


# ====== KONFIG ======
DATASET_ROOT = Path(r"C:\Users\dyksi\OneDrive\Pulpit\magisterka\yolo_test\trening\dataset_v8_linie_500")

# 0 = background, 1 = red_line, 2 = white_line, 3 = yellow_line
NUM_CLASSES = 4
CLASS_NAMES = ["background", "red_line", "white_line", "yellow_line"]

IMG_SIZE = 512
BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 50
BASE_CHANNELS = 32
NUM_WORKERS = 0
SAVE_BEST_BY = "iou"   # "iou" albo "dice"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

RUNS_ROOT = Path(__file__).resolve().parent / "runs_unet"
RUN_NAME = "train"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


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
    cfg.append(f"BASE_CHANNELS={BASE_CHANNELS}")
    cfg.append(f"NUM_WORKERS={NUM_WORKERS}")
    cfg.append(f"SAVE_BEST_BY={SAVE_BEST_BY}")
    cfg.append(f"DEVICE={DEVICE}")
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
    # cm: rows=true, cols=pred
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
            if normalize:
                txt = f"{mat[i, j]:.2f}"
            else:
                txt = f"{int(cm[i, j])}"
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
    """
    Prosta wizualizacja klas do PNG.
    """
    palette = np.array([
        [0, 0, 0],        # background
        [255, 0, 0],      # red_line
        [255, 255, 255],  # white_line
        [255, 255, 0],    # yellow_line
        [0, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128],
    ], dtype=np.uint8)

    if num_classes > len(palette):
        raise ValueError("Za mała paleta kolorów dla liczby klas.")

    return palette[mask]


def save_prediction_examples(model, dataset, device, out_dir: Path, num_examples: int = 6):
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    n = min(num_examples, len(dataset))

    with torch.no_grad():
        for idx in range(n):
            img_t, mask_t = dataset[idx]

            logits = model(img_t.unsqueeze(0).to(device))
            pred_t = torch.argmax(logits, dim=1)[0].cpu().numpy()

            img = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            gt = mask_t.cpu().numpy().astype(np.uint8)
            pred = pred_t.astype(np.uint8)

            gt_vis = colorize_mask(gt, NUM_CLASSES)
            pred_vis = colorize_mask(pred, NUM_CLASSES)

            canvas = np.concatenate([img, gt_vis, pred_vis], axis=1)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(out_dir / f"sample_{idx:02d}.png"), canvas)


# ====== DATASET ======
class SegDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transform=None, num_classes=4):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.num_classes = num_classes

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
            preview = "\n".join(str(p) for p in missing_masks[:10])
            more = f"\n... i jeszcze {len(missing_masks) - 10} brakujących masek" if len(missing_masks) > 10 else ""
            raise RuntimeError(
                f"Brakuje {len(missing_masks)} masek w {self.masks_dir}.\n"
                f"Przykłady:\n{preview}{more}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Nie mogę wczytać obrazu: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Nie mogę wczytać maski: {mask_path}")

        if img.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"Rozmiar obrazu i maski się różni dla {img_path.name}: "
                f"img={img.shape[:2]}, mask={mask.shape[:2]}"
            )

        max_val = int(mask.max())
        if max_val >= self.num_classes:
            raise RuntimeError(
                f"Maska ma wartość {max_val} >= NUM_CLASSES={self.num_classes} w {mask_path}"
            )

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return img, mask


# ====== TRANSFORMACJE ======
train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])


# ====== PROSTY UNET ======
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


# ====== TRAIN / VAL ======
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    n = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for imgs, masks in tqdm(loader, leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = ce(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        update_confusion_matrix(cm, preds, masks, NUM_CLASSES)

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
    run_dir = make_run_dir(RUNS_ROOT, RUN_NAME)
    print(f"[INFO] Run dir: {run_dir}")
    save_config(run_dir)

    train_images = DATASET_ROOT / "train" / "images"
    train_masks = DATASET_ROOT / "train" / "masks"
    val_images = DATASET_ROOT / "val" / "images"
    val_masks = DATASET_ROOT / "val" / "masks"

    train_ds = SegDataset(train_images, train_masks, transform=train_tf, num_classes=NUM_CLASSES)
    val_ds = SegDataset(val_images, val_masks, transform=val_tf, num_classes=NUM_CLASSES)

    print(f"[INFO] Train: {len(train_ds)} obrazów")
    print(f"[INFO] Val:   {len(val_ds)} obrazów")
    print(f"[INFO] Device: {DEVICE}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base=BASE_CHANNELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    best_score = -1.0
    best_epoch = -1
    best_cm = None
    best_iou_per_class = None
    best_dice_per_class = None

    best_weights_path = run_dir / "best.pt"
    last_weights_path = run_dir / "last.pt"
    results_csv_path = run_dir / "results.csv"

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None)

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

        torch.save(model.state_dict(), last_weights_path)

        score = row["val_iou"] if SAVE_BEST_BY == "iou" else row["val_dice"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_cm = val_metrics["cm"].copy()
            best_iou_per_class = val_metrics["iou_per_class"].copy()
            best_dice_per_class = val_metrics["dice_per_class"].copy()

            torch.save(model.state_dict(), best_weights_path)
            print(f"  [SAVE] best.pt (best val {SAVE_BEST_BY} = {best_score:.4f})")

            examples_dir = run_dir / "val_examples"
            if examples_dir.exists():
                shutil.rmtree(examples_dir)
            save_prediction_examples(model, val_ds, DEVICE, examples_dir, num_examples=6)

        plot_curve(history, "train_loss", "val_loss", "Loss", run_dir / "loss.png")
        plot_curve(history, "train_iou", "val_iou", "IoU", run_dir / "iou.png")
        plot_curve(history, "train_dice", "val_dice", "Dice", run_dir / "dice.png")

    total_training_time = time.time() - start_time

    print(f"\n[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best val {SAVE_BEST_BY}: {best_score:.4f}")
    print(f"[INFO] Łączny czas treningu: {total_training_time:.2f} s")

    # zapis confusion matrix i metryk per class dla best epoch
    if best_cm is not None:
        save_confusion_matrix_image(best_cm, CLASS_NAMES, run_dir / "confusion_matrix_raw.png", normalize=False)
        save_confusion_matrix_image(best_cm, CLASS_NAMES, run_dir / "confusion_matrix_norm.png", normalize=True)
        save_per_class_metrics_csv(CLASS_NAMES, best_iou_per_class, best_dice_per_class, run_dir / "per_class_metrics.csv")

    print(f"[INFO] Zapisano do: {run_dir}")
    print("[INFO] Pliki:")
    print("       - best.pt")
    print("       - last.pt")
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
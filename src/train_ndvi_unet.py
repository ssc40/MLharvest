import os
import json
from datetime import datetime
from glob import glob
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

# =========================
# CONFIG
# =========================
RGB_DIR  = "imageDatasetwithDates/content/True_Color_Data"
NDVI_DIR = "imageDatasetwithDates/content/NDVI_Data"
RESULTS_DIR = "results"

IMG_SHAPE  = (496, 512)   # (H, W) — multiples of 16 required by UNet pooling
BATCH_SIZE = 2
EPOCHS     = 50
LR         = 1e-4
VAL_SPLIT  = 0.15   # fraction held out for validation
TEST_SPLIT = 0.15   # fraction held out for final evaluation (never seen during training)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# =========================
# DATASET
# =========================
class NDVIDataset(Dataset):
    def __init__(self, rgb_dir, ndvi_dir, img_shape=IMG_SHAPE, augment=False):
        all_rgb = sorted(glob(os.path.join(rgb_dir, "*.png")))
        # Keep only pairs that have a matching NDVI file
        self.rgb_paths = [
            p for p in all_rgb
            if os.path.exists(os.path.join(ndvi_dir, os.path.basename(p)))
        ]
        self.ndvi_dir  = ndvi_dir
        self.img_shape = img_shape
        self.augment   = augment

        self.transform_rgb = T.Compose([
            T.Resize(self.img_shape),
            T.ToTensor(),   # uint8 → float32 in [0, 1]
        ])
        self.transform_ndvi = T.Compose([
            T.Resize(self.img_shape, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),   # uint8 → float32 in [0, 1]
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path  = self.rgb_paths[idx]
        ndvi_path = os.path.join(self.ndvi_dir, os.path.basename(rgb_path))

        rgb  = Image.open(rgb_path).convert("RGB")
        ndvi = Image.open(ndvi_path).convert("L")   # grayscale, 1 channel

        rgb  = self.transform_rgb(rgb)
        ndvi = self.transform_ndvi(ndvi)

        # RGB: [0, 1] → [-1, 1]  (Sentinel-like normalisation)
        rgb = rgb * 2.0 - 1.0   # type: ignore[operator]

        # NDVI PNGs are encoded as uint8 where 0 → NDVI=-1 and 255 → NDVI=+1.
        # T.ToTensor() gives [0, 1]; rescale back to [-1, 1] to match tanh output.
        ndvi = ndvi * 2.0 - 1.0  # type: ignore[operator]

        # Paired augmentation — same random flip applied to both tensors
        if self.augment:
            if torch.rand(1).item() > 0.5:
                rgb  = torch.flip(rgb,  dims=[2])   # type: ignore[arg-type]
                ndvi = torch.flip(ndvi, dims=[2])   # type: ignore[arg-type]  # horizontal
            if torch.rand(1).item() > 0.5:
                rgb  = torch.flip(rgb,  dims=[1])   # type: ignore[arg-type]
                ndvi = torch.flip(ndvi, dims=[1])   # type: ignore[arg-type]  # vertical

        return rgb, ndvi


# =========================
# MODEL
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.tanh(self.final(d1))   # output in [-1, 1]


# =========================
# HELPERS
# =========================
def _make_run_dir() -> str:
    """Create a timestamped run directory under RESULTS_DIR and return its path.

    Layout:
        results/
            run_20260503_143022/
                hparams.json          ← hyperparameters
                best.pth              ← best val-MSE checkpoint
                last.pth              ← final-epoch checkpoint
                training_metrics.png  ← loss/metric curves
                sample_predictions.png
    """
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_hparams(run_dir: str, extra: dict | None = None) -> None:
    """Dump all CONFIG constants plus any extra values to hparams.json."""
    hparams = {
        "rgb_dir":     RGB_DIR,
        "ndvi_dir":    NDVI_DIR,
        "img_shape":   list(IMG_SHAPE),
        "batch_size":  BATCH_SIZE,
        "epochs":      EPOCHS,
        "lr":          LR,
        "val_split":   VAL_SPLIT,
        "test_split":  TEST_SPLIT,
        "device":      DEVICE,
        "augment":     True,
        "scheduler":   "ReduceLROnPlateau(factor=0.5, patience=4)",
    }
    if extra:
        hparams.update(extra)
    path = os.path.join(run_dir, "hparams.json")
    with open(path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"Hyperparameters saved to {path}")


def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Epoch-level R². Pass the full concatenated prediction/target tensors."""
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def _ndvi_to_colormap(ndvi_tensor):
    """
    Convert a (1, H, W) or (H, W) NDVI tensor in [-1, 1] to a (H, W, 3)
    uint8 numpy array using the RdYlGn colormap (red=bare/stressed,
    yellow=moderate, green=dense vegetation).
    """
    arr = ndvi_tensor.squeeze().cpu().float().numpy()   # (H, W)
    arr = (arr + 1.0) / 2.0                             # [0, 1] for colormap
    colored = matplotlib.colormaps["RdYlGn"](arr)[:, :, :3]  # (H, W, 3) float [0, 1]
    return (colored * 255).astype(np.uint8)


def _save_sample_predictions(model, dataset, indices, save_path):
    """Save a side-by-side figure: RGB | Predicted NDVI | Ground-Truth NDVI."""
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            rgb_t, ndvi_gt = dataset[idx]
            pred = model(rgb_t.unsqueeze(0).to(DEVICE))[0]   # (1, H, W)

            rgb_disp = ((rgb_t + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
            rgb_disp = np.clip(rgb_disp, 0, 1)

            axes[row, 0].imshow(rgb_disp)
            axes[row, 0].set_title(f"RGB Input  (sample {idx})")

            axes[row, 1].imshow(_ndvi_to_colormap(pred.cpu()))
            axes[row, 1].set_title("Predicted NDVI")

            axes[row, 2].imshow(_ndvi_to_colormap(ndvi_gt))
            axes[row, 2].set_title("Ground-Truth NDVI")

            for ax in axes[row]:
                ax.axis("off")

    # Shared colorbar
    sm = mcm.ScalarMappable(cmap="RdYlGn", norm=mcolors.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:, 1:].ravel().tolist(), shrink=0.6, label="NDVI")

    plt.suptitle("NDVI Predictions vs Ground Truth", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved to {save_path}")


def _save_plots(history, save_dir):
    epochs = range(1, len(history["train_mse"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("NDVI U-Net Training Metrics", fontsize=14)

    metrics = [
        ("MSE Loss",  "train_mse",  "val_mse"),
        ("MAE",       "train_mae",  "val_mae"),
        ("RMSE",      "train_rmse", "val_rmse"),
        ("R²",        "train_r2",   "val_r2"),
    ]
    for ax, (title, train_key, val_key) in zip(axes.flat, metrics):
        ax.plot(epochs, history[train_key], label="Train", marker="o")
        ax.plot(epochs, history[val_key],   label="Val",   marker="o")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Metrics plot saved to {out}")


def _run_eval(model, loader, mse_fn, mae_fn, split="Test"):
    model.eval()
    mse = mae = 0.0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for rgb, ndvi in loader:
            rgb, ndvi = rgb.to(DEVICE), ndvi.to(DEVICE)
            pred = model(rgb)
            mse += mse_fn(pred, ndvi).item()
            mae += mae_fn(pred, ndvi).item()
            all_preds.append(pred.cpu())
            all_targets.append(ndvi.cpu())
    n = len(loader)
    epoch_r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets))
    results = {"mse": mse / n, "mae": mae / n, "rmse": (mse / n) ** 0.5, "r2": epoch_r2}
    print(
        f"{split} MSE:  {results['mse']:.6f}\n"
        f"{split} MAE:  {results['mae']:.6f}\n"
        f"{split} RMSE: {results['rmse']:.6f}\n"
        f"{split} R²:   {results['r2']:.4f}"
    )
    return results


# =========================
# TRAINING
# =========================
def train():
    import tqdm
    from torch.utils.data import random_split

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_dir = _make_run_dir()
    print(f"Run directory: {run_dir}")

    # Build a base dataset (no augmentation) to determine the split indices
    dataset = NDVIDataset(RGB_DIR, NDVI_DIR, IMG_SHAPE, augment=False)
    n_test  = max(1, int(len(dataset) * TEST_SPLIT))
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap the train subset with augmentation enabled
    aug_dataset   = NDVIDataset(RGB_DIR, NDVI_DIR, IMG_SHAPE, augment=True)
    train_set_aug = torch.utils.data.Subset(aug_dataset, train_set.indices)

    train_loader = DataLoader(train_set_aug, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")

    _save_hparams(run_dir, extra={"n_train": n_train, "n_val": n_val, "n_test": n_test})

    # Pick a few val indices for visualisation (indices into the original dataset)
    vis_indices = list(val_set.indices[: min(3, n_val)])

    model     = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    history = {
        "train_mse": [], "val_mse": [],
        "train_mae": [], "val_mae": [],
        "train_rmse": [], "val_rmse": [],
        "train_r2":  [], "val_r2":  [],
    }

    best_val_mse = float("inf")
    best_ckpt    = os.path.join(run_dir, "best.pth")

    for epoch in tqdm.trange(EPOCHS):
        # --- training ---
        model.train()
        t_mse = t_mae = 0.0
        t_preds: list[torch.Tensor] = []
        t_targets: list[torch.Tensor] = []
        for rgb, ndvi in train_loader:
            rgb, ndvi = rgb.to(DEVICE), ndvi.to(DEVICE)
            pred = model(rgb)
            loss = mse_fn(pred, ndvi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_mse += loss.item()
            t_mae += mae_fn(pred, ndvi).item()
            t_preds.append(pred.detach().cpu())
            t_targets.append(ndvi.cpu())

        n = len(train_loader)
        train_r2 = r2_score(torch.cat(t_preds), torch.cat(t_targets))
        history["train_mse"].append(t_mse / n)
        history["train_mae"].append(t_mae / n)
        history["train_rmse"].append((t_mse / n) ** 0.5)
        history["train_r2"].append(train_r2)

        # --- validation ---
        model.eval()
        v_mse = v_mae = 0.0
        v_preds: list[torch.Tensor] = []
        v_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for rgb, ndvi in val_loader:
                rgb, ndvi = rgb.to(DEVICE), ndvi.to(DEVICE)
                pred = model(rgb)
                v_mse += mse_fn(pred, ndvi).item()
                v_mae += mae_fn(pred, ndvi).item()
                v_preds.append(pred.cpu())
                v_targets.append(ndvi.cpu())

        m = len(val_loader)
        val_r2 = r2_score(torch.cat(v_preds), torch.cat(v_targets))
        cur_val_mse = v_mse / m
        history["val_mse"].append(cur_val_mse)
        history["val_mae"].append(v_mae / m)
        history["val_rmse"].append((v_mse / m) ** 0.5)
        history["val_r2"].append(val_r2)

        scheduler.step(cur_val_mse)

        # Save best checkpoint
        if cur_val_mse < best_val_mse:
            best_val_mse = cur_val_mse
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train MSE: {history['train_mse'][-1]:.6f}  Val MSE: {cur_val_mse:.6f}  "
            f"Train MAE: {history['train_mae'][-1]:.6f}  Val MAE: {history['val_mae'][-1]:.6f}  "
            f"Val R\u00b2: {val_r2:.4f}"
        )

    # Save last weights
    last_ckpt = os.path.join(run_dir, "last.pth")
    torch.save(model.state_dict(), last_ckpt)
    print(f"\nLast model  \u2192 {last_ckpt}")
    print(f"Best model  \u2192 {best_ckpt}  (val MSE {best_val_mse:.6f})")

    _save_plots(history, run_dir)

    # Visualise sample predictions from the best model
    model.load_state_dict(
        torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
    )
    _save_sample_predictions(
        model, dataset, vis_indices,
        os.path.join(run_dir, "sample_predictions.png"),
    )

    # Write final test metrics into hparams.json so everything is in one place
    print("\n--- Final evaluation on held-out test set (best model) ---")
    test_results = _run_eval(model, test_loader, mse_fn, mae_fn, split="Test")
    hparams_path = os.path.join(run_dir, "hparams.json")
    with open(hparams_path) as f:
        hparams = json.load(f)
    hparams["test_results"] = test_results
    hparams["best_val_mse"] = best_val_mse
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"\nAll run artefacts saved to: {run_dir}")


# =========================
# EVALUATE
# =========================
def evaluate(model_path, rgb_dir=RGB_DIR, ndvi_dir=NDVI_DIR):
    """Load saved weights and report metrics on the held-out test split."""
    from torch.utils.data import random_split

    dataset = NDVIDataset(rgb_dir, ndvi_dir, IMG_SHAPE)
    n_test  = max(1, int(len(dataset) * TEST_SPLIT))
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val - n_test
    _, _, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)   # same seed → same split
    )
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f"Loaded model from {model_path}  |  Test samples: {n_test}")

    _run_eval(model, test_loader, nn.MSELoss(), nn.L1Loss(), split="Test")


# =========================
# PREDICT (inference + visualisation)
# =========================
def predict(model_path, rgb_image_path, output_path=None, ndvi_gt_path=None):
    """
    Run the model on a single RGB image and save an NDVI map.

    Args:
        model_path:      Path to saved model weights (.pth).
        rgb_image_path:  Path to the input RGB image (PNG/JPG, any size).
        output_path:     Where to save the output PNG.  Defaults to
                         results/<stem>_ndvi.png.
        ndvi_gt_path:    Optional path to a ground-truth NDVI image for a
                         side-by-side comparison panel.

    Returns:
        ndvi_np: (H, W) numpy array with NDVI values in [-1, 1].
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if output_path is None:
        stem = Path(rgb_image_path).stem
        output_path = os.path.join(RESULTS_DIR, f"{stem}_ndvi.png")

    transform = T.Compose([T.Resize(IMG_SHAPE), T.ToTensor()])

    rgb_pil = Image.open(rgb_image_path).convert("RGB")
    rgb_t   = transform(rgb_pil) * 2.0 - 1.0   # type: ignore[operator]  # (3, H, W) in [-1, 1]
    rgb_in  = rgb_t.unsqueeze(0).to(DEVICE)           # (1, 3, H, W)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(rgb_in)[0]   # (1, H, W)

    ndvi_np = pred.squeeze().cpu().numpy()             # (H, W) in [-1, 1]

    # ---- Build visualisation ----
    has_gt = ndvi_gt_path is not None and os.path.exists(ndvi_gt_path)
    ncols  = 3 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    rgb_disp = np.clip(((rgb_t + 1.0) / 2.0).permute(1, 2, 0).numpy(), 0, 1)
    axes[0].imshow(rgb_disp)
    axes[0].set_title("RGB Input")

    axes[1].imshow(_ndvi_to_colormap(pred.cpu()))
    axes[1].set_title("Predicted NDVI")

    if has_gt and ndvi_gt_path is not None:
        gt_t = transform(Image.open(ndvi_gt_path).convert("L")) * 2.0 - 1.0  # type: ignore[operator]
        axes[2].imshow(_ndvi_to_colormap(gt_t))
        axes[2].set_title("Ground-Truth NDVI")

    # Shared colorbar
    sm = mcm.ScalarMappable(cmap="RdYlGn", norm=mcolors.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.04, label="NDVI")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(Path(rgb_image_path).stem, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"NDVI map saved to {output_path}")
    return ndvi_np


# =========================
# AUGMENTATION CHECK
# =========================
def check_augment(n_draws: int = 4, sample_idx: int = 0):
    """Sample one image n_draws times from the augmented dataset and save a
    grid so you can visually confirm flips are being applied.

    Usage:
        python src/train_ndvi_unet.py check-augment
        python src/train_ndvi_unet.py check-augment --n 6 --idx 10
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    aug_ds = NDVIDataset(RGB_DIR, NDVI_DIR, IMG_SHAPE, augment=True)
    n_ds = len(aug_ds)
    if sample_idx >= n_ds:
        raise ValueError(f"--idx {sample_idx} is out of range (dataset has {n_ds} samples)")

    fig, axes = plt.subplots(2, n_draws, figsize=(4 * n_draws, 8))
    fig.suptitle(
        f"Augmentation check — sample {sample_idx}, {n_draws} draws\n"
        "(horizontal/vertical flips applied randomly per draw)",
        fontsize=11,
    )

    for col in range(n_draws):
        rgb_t, ndvi_t = aug_ds[sample_idx]
        rgb_disp = np.clip(((rgb_t + 1.0) / 2.0).permute(1, 2, 0).numpy(), 0, 1)
        axes[0, col].imshow(rgb_disp)
        axes[0, col].set_title(f"RGB draw {col + 1}")
        axes[0, col].axis("off")

        axes[1, col].imshow(_ndvi_to_colormap(ndvi_t))
        axes[1, col].set_title(f"NDVI draw {col + 1}")
        axes[1, col].axis("off")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"augment_check_idx{sample_idx}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Augmentation check saved to {out}")
    print(
        "If augmentation is working you should see different orientations "
        "across the columns (50 % chance of each flip per draw)."
    )


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NDVI U-Net — train, evaluate, or predict")
    sub = parser.add_subparsers(dest="cmd")

    # train
    sub.add_parser("train", help="Train from scratch")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate saved weights on the test split")
    p_eval.add_argument("model_path", help="Path to .pth weights file")

    # predict
    p_pred = sub.add_parser("predict", help="Run inference on a single RGB image")
    p_pred.add_argument("model_path",     help="Path to .pth weights file")
    p_pred.add_argument("rgb_image_path", help="Path to input RGB image")
    p_pred.add_argument("--gt",    dest="ndvi_gt_path", default=None,
                        help="Optional ground-truth NDVI image for comparison")
    p_pred.add_argument("--output", dest="output_path", default=None,
                        help="Where to save the output PNG (default: results/<stem>_ndvi.png)")

    # check-augment
    p_aug = sub.add_parser("check-augment", help="Visually verify augmentation is working")
    p_aug.add_argument("--n",   type=int, default=4,  dest="n_draws",    help="Number of draws (default: 4)")
    p_aug.add_argument("--idx", type=int, default=0,  dest="sample_idx", help="Dataset index to sample (default: 0)")

    args = parser.parse_args()

    if args.cmd == "evaluate":
        evaluate(args.model_path)
    elif args.cmd == "predict":
        predict(args.model_path, args.rgb_image_path,
                output_path=args.output_path, ndvi_gt_path=args.ndvi_gt_path)
    elif args.cmd == "check-augment":
        check_augment(n_draws=args.n_draws, sample_idx=args.sample_idx)
    else:
        train()   # default: train
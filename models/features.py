"""
Shared feature extraction for VIGIL-CV's demo model + embedding-based detectors.

Kept deliberately simple and dependency-light (numpy + PIL only) so the whole
pipeline runs on CPU in seconds with no heavyweight deep-learning framework.
The "embedding" doubles as: (a) the input to the classifier, and (b) the
embedding space used by F1's OOD / near-duplicate / label-consistency checks.
"""
import numpy as np
from PIL import Image

IMG_SIZE = 32
CLASSES = ["circle", "square", "triangle"]


def load_image_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0


def flatten_features(img_arr: np.ndarray) -> np.ndarray:
    """Flattened, normalized pixel vector -> classifier input (3072-dim)."""
    return img_arr.reshape(-1).astype(np.float32)


def embedding(img_arr: np.ndarray) -> np.ndarray:
    """
    Cheap but meaningful embedding for similarity/OOD/duplicate checks:
    downsampled color grid + global color/texture stats.
    Lower-dimensional than the raw pixels so distance metrics behave sanely.
    """
    small = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((8, 8))
    grid = (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1)  # 192-d
    mean_rgb = img_arr.reshape(-1, 3).mean(axis=0)
    std_rgb = img_arr.reshape(-1, 3).std(axis=0)
    gray = img_arr.mean(axis=2)
    edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    return np.concatenate([grid, mean_rgb, std_rgb, [edges]]).astype(np.float32)


def foreground_embedding(img_arr: np.ndarray, bg_thresh: float = 0.82) -> np.ndarray:
    """
    Class-discriminative embedding used by label-consistency / OOD checks.
    Isolates the drawn shape from the near-white background (by simple
    brightness thresholding) so background-noise variation doesn't swamp the
    color/shape signal that actually distinguishes classes -- a background-
    uniform grid embedding is dominated by the background, which is exactly
    the failure mode a naive detector would have.
    """
    mask = img_arr.min(axis=2) < bg_thresh  # foreground = not-near-white
    if mask.sum() < 5:
        fg_mean = img_arr.reshape(-1, 3).mean(axis=0)
        fg_std = np.zeros(3, dtype=np.float32)
        fg_frac = 0.0
        cy, cx = 0.5, 0.5
    else:
        fg_pixels = img_arr[mask]
        fg_mean = fg_pixels.mean(axis=0)
        fg_std = fg_pixels.std(axis=0)
        fg_frac = float(mask.sum()) / mask.size
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean() / mask.shape[0], xs.mean() / mask.shape[1]
    return np.concatenate([fg_mean, fg_std, [fg_frac, cy, cx]]).astype(np.float32)


def phash(img_arr: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Simple perceptual hash (DCT-free average-hash) for near-duplicate detection."""
    gray = Image.fromarray((img_arr * 255).astype(np.uint8)).convert("L").resize(
        (hash_size, hash_size)
    )
    arr = np.asarray(gray, dtype=np.float32)
    return (arr > arr.mean()).astype(np.uint8).reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))

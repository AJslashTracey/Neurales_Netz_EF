import os

import matplotlib.pyplot as plt
import numpy as np

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_images_from_folder(folder_path: str) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(10):
        digit_path = os.path.join(folder_path, str(digit))
        files = sorted(
            f for f in os.listdir(digit_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)
        )
        for filename in files:
            img = plt.imread(os.path.join(digit_path, filename))
            img = img.astype(np.float32)
            if img.ndim == 3:
                # Keep compatibility with possible RGB/RGBA inputs.
                img = img[..., 0]
            max_val = float(np.max(img))
            if max_val > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0).flatten()
            images.append(img)
            labels.append(digit)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

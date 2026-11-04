
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "watershed_objects.jpg"

OUTPUT_DIR = BASE_DIR / "outputs" / "task1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD IMAGE
# --------------------------------------------------

image = cv2.imread(str(INPUT_PATH))

if image is None:
    raise FileNotFoundError(
        f"Image not found: {INPUT_PATH}"
    )

gray = cv2.cvtColor(
    image, cv2.COLOR_BGR2GRAY
)

print("Watershed image loaded successfully!")
print("Image dimensions:", image.shape)


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# --------------------------------------------------
# 4. AUTOMATIC THRESHOLDING
# --------------------------------------------------

# Check background brightness using image borders
h, w = gray.shape

border_size = max(3, min(h, w) // 20)

border_pixels = np.concatenate([
    gray[:border_size, :].ravel(),
    gray[-border_size:, :].ravel(),
    gray[:, :border_size].ravel(),
    gray[:, -border_size:].ravel()
])

# Select threshold direction
if np.mean(border_pixels) > np.mean(gray):
    threshold_type = cv2.THRESH_BINARY_INV
else:
    threshold_type = cv2.THRESH_BINARY

_, binary = cv2.threshold(
    blurred,
    0,
    255,
    threshold_type + cv2.THRESH_OTSU
)

cv2.imwrite(
    str(OUTPUT_DIR / "watershed_binary.jpg"),
    binary
)


# --------------------------------------------------
# 5. REMOVE SMALL NOISE
# --------------------------------------------------

kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)


# --------------------------------------------------
# 6. FIND SURE BACKGROUND
# --------------------------------------------------

sure_background = cv2.dilate(
    opening,
    kernel,
    iterations=3
)


# --------------------------------------------------
# 7. FIND SURE FOREGROUND
# --------------------------------------------------

distance = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

_, sure_foreground = cv2.threshold(
    distance,
    0.35 * distance.max(),
    255,
    cv2.THRESH_BINARY
)

sure_foreground = np.uint8(
    sure_foreground
)

cv2.imwrite(
    str(OUTPUT_DIR / "watershed_foreground.jpg"),
    sure_foreground
)


# --------------------------------------------------
# 8. FIND UNKNOWN REGION
# --------------------------------------------------

unknown = cv2.subtract(
    sure_background,
    sure_foreground
)


# --------------------------------------------------
# 9. CREATE MARKERS
# --------------------------------------------------

num_labels, markers = cv2.connectedComponents(
    sure_foreground
)

print("\nNumber of foreground marker regions:", num_labels - 1)

markers = markers + 1

markers[unknown == 255] = 0


# --------------------------------------------------
# 10. APPLY WATERSHED
# --------------------------------------------------

markers = cv2.watershed(
    image.copy(),
    markers
)


# --------------------------------------------------
# 11. DRAW SEGMENTATION BOUNDARIES
# --------------------------------------------------

watershed_result = image.copy()

# Red indicates watershed boundaries
watershed_result[markers == -1] = [0, 0, 255]

cv2.imwrite(
    str(OUTPUT_DIR / "watershed_result_coins.jpg"),
    watershed_result
)


# --------------------------------------------------
# 12. CREATE COLOURED SEGMENTATION
# --------------------------------------------------

segmented = np.zeros_like(image)

rng = np.random.default_rng(42)

for label in np.unique(markers):

    if label <= 1:
        continue

    mask = markers == label

    color = rng.integers(
        50, 255, size=3, dtype=np.uint8
    )

    segmented[mask] = color

cv2.imwrite(
    str(OUTPUT_DIR / "watershed_colored_segments.jpg"),
    segmented
)


# --------------------------------------------------
# 13. DISPLAY AND SAVE COMPARISON
# --------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

axes[0, 0].imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)
axes[0, 0].set_title("Original Coins Image")

axes[0, 1].imshow(
    binary,
    cmap="gray"
)
axes[0, 1].set_title("Binary Thresholding")

axes[1, 0].imshow(
    sure_foreground,
    cmap="gray"
)
axes[1, 0].set_title("Sure Foreground")

axes[1, 1].imshow(
    cv2.cvtColor(
        watershed_result,
        cv2.COLOR_BGR2RGB
    )
)
axes[1, 1].set_title("Final Watershed Segmentation")

for ax in axes.flatten():
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "watershed_coins_summary.png",
    dpi=150
)

plt.show()

print("\nWATERSHED SEGMENTATION COMPLETED!")
print("Results saved inside:", OUTPUT_DIR)

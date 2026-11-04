
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "original_photo.jpeg"
OUTPUT_DIR = BASE_DIR / "outputs" / "task1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Read original photograph
original = cv2.imread(str(INPUT_PATH))

if original is None:
    raise FileNotFoundError(
        f"Cannot find photograph: {INPUT_PATH}"
    )

height, width = original.shape[:2]

print("Original image loaded successfully!")
print("Image dimensions:", width, "x", height)


# --------------------------------------------------
# 1. ADD GAUSSIAN NOISE
# --------------------------------------------------

noise = np.random.normal(
    0, 25, original.shape
).astype(np.float32)

gaussian_noise = np.clip(
    original.astype(np.float32) + noise,
    0, 255
).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT_DIR / "gaussian_noise.jpg"),
    gaussian_noise
)


# --------------------------------------------------
# 2. ADD SALT AND PEPPER NOISE
# --------------------------------------------------

salt_pepper = original.copy()

random_values = np.random.random((height, width))

salt_pepper[random_values < 0.02] = 0
salt_pepper[random_values > 0.98] = 255

cv2.imwrite(
    str(OUTPUT_DIR / "salt_pepper_noise.jpg"),
    salt_pepper
)


# --------------------------------------------------
# 3. IMAGE RESTORATION
# --------------------------------------------------

# Median filtering
gaussian_median = cv2.medianBlur(
    gaussian_noise, 5
)

salt_pepper_median = cv2.medianBlur(
    salt_pepper, 5
)

# Gaussian filtering
gaussian_filtered = cv2.GaussianBlur(
    gaussian_noise, (5, 5), 0
)

salt_pepper_gaussian = cv2.GaussianBlur(
    salt_pepper, (5, 5), 0
)


# Save restored images
images = {
    "gaussian_noise_median.jpg": gaussian_median,
    "salt_pepper_median.jpg": salt_pepper_median,
    "gaussian_noise_gaussian.jpg": gaussian_filtered,
    "salt_pepper_gaussian.jpg": salt_pepper_gaussian
}

for filename, image in images.items():
    cv2.imwrite(str(OUTPUT_DIR / filename), image)


# --------------------------------------------------
# 4. PSNR AND SSIM CALCULATION
# --------------------------------------------------

def calculate_metrics(reference, result):

    psnr = peak_signal_noise_ratio(
        reference, result, data_range=255
    )

    ssim = structural_similarity(
        reference,
        result,
        channel_axis=2,
        data_range=255
    )

    return psnr, ssim


print("\nIMAGE QUALITY COMPARISON")

results = {
    "Gaussian Noise": gaussian_noise,
    "Gaussian + Median Filter": gaussian_median,
    "Gaussian + Gaussian Filter": gaussian_filtered,
    "Salt & Pepper Noise": salt_pepper,
    "Salt & Pepper + Median Filter": salt_pepper_median,
    "Salt & Pepper + Gaussian Filter": salt_pepper_gaussian
}

for name, image in results.items():

    psnr, ssim = calculate_metrics(
        original, image
    )

    print(f"\n{name}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


# --------------------------------------------------
# 5. THRESHOLDING
# --------------------------------------------------

gray = cv2.cvtColor(
    original, cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite(
    str(OUTPUT_DIR / "binary_threshold.jpg"),
    binary
)


# --------------------------------------------------
# 6. MORPHOLOGICAL OPERATIONS
# --------------------------------------------------

kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(
    binary, kernel, iterations=1
)

# Dilation
dilation = cv2.dilate(
    binary, kernel, iterations=1
)

# Opening
opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

# Closing
closing = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, kernel
)


cv2.imwrite(str(OUTPUT_DIR / "erosion.jpg"), erosion)
cv2.imwrite(str(OUTPUT_DIR / "dilation.jpg"), dilation)
cv2.imwrite(str(OUTPUT_DIR / "opening.jpg"), opening)
cv2.imwrite(str(OUTPUT_DIR / "closing.jpg"), closing)


# --------------------------------------------------
# 7. CANNY EDGE DETECTION
# --------------------------------------------------

edges = cv2.Canny(
    gray, 100, 200
)

cv2.imwrite(
    str(OUTPUT_DIR / "canny_edges.jpg"),
    edges
)


# --------------------------------------------------
# 8. WATERSHED SEGMENTATION
# --------------------------------------------------

# Prepare foreground and background markers
watershed_image = original.copy()

# Use a binary foreground mask
_, mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Remove small noise
opening_mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

# Sure background
sure_background = cv2.dilate(
    opening_mask, kernel, iterations=3
)

# Sure foreground using distance transform
distance = cv2.distanceTransform(
    opening_mask, cv2.DIST_L2, 5
)

_, sure_foreground = cv2.threshold(
    distance,
    0.35 * distance.max(),
    255,
    0
)

sure_foreground = np.uint8(sure_foreground)

# Unknown region
unknown = cv2.subtract(
    sure_background, sure_foreground
)

# Create markers
_, markers = cv2.connectedComponents(
    sure_foreground
)

markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(
    watershed_image, markers
)

# Mark watershed boundaries in red
watershed_image[markers == -1] = [0, 0, 255]

cv2.imwrite(
    str(OUTPUT_DIR / "watershed_result.jpg"),
    watershed_image
)


# --------------------------------------------------
# 9. DISPLAY ALL IMPORTANT RESULTS
# --------------------------------------------------

display_images = [
    ("Original", original),
    ("Gaussian Noise", gaussian_noise),
    ("Salt & Pepper Noise", salt_pepper),
    ("Median Filter", gaussian_median),
    ("Gaussian Filter", gaussian_filtered),
    ("Binary Image", binary),
    ("Erosion", erosion),
    ("Dilation", dilation),
    ("Opening", opening),
    ("Closing", closing),
    ("Canny Edges", edges),
    ("Watershed", watershed_image)
]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for ax, (title, image) in zip(
    axes.flatten(), display_images
):

    if len(image.shape) == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task1_summary.png",
    dpi=150
)

plt.show()

print("\nTASK 1 COMPLETED SUCCESSFULLY!")
print("All outputs saved inside:", OUTPUT_DIR)

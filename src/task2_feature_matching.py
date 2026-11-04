
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.feature import hog


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE1_PATH = BASE_DIR / "dataset" / "second_photo.jpeg"
IMAGE2_PATH = BASE_DIR / "dataset" / "third_photo.jpeg"

OUTPUT_DIR = BASE_DIR / "outputs" / "task2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD BOTH PHOTOGRAPHS
# --------------------------------------------------

image1 = cv2.imread(str(IMAGE1_PATH))
image2 = cv2.imread(str(IMAGE2_PATH))

if image1 is None:
    raise FileNotFoundError(f"Image not found: {IMAGE1_PATH}")

if image2 is None:
    raise FileNotFoundError(f"Image not found: {IMAGE2_PATH}")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

print("Both photographs loaded successfully!")
print("Image 1 dimensions:", image1.shape)
print("Image 2 dimensions:", image2.shape)


# --------------------------------------------------
# 3. HOG FEATURE EXTRACTION
# --------------------------------------------------

def extract_hog(image, name):

    # Resize image for consistent HOG extraction
    resized = cv2.resize(image, (256, 256))

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values
    gray = gray.astype(np.float32) / 255.0

    # Extract HOG features
    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    # Save feature vector
    np.save(
        OUTPUT_DIR / f"{name}_hog_features.npy",
        features
    )

    return gray, features, hog_image


gray_hog1, features1, hog_image1 = extract_hog(
    image1, "image1"
)

gray_hog2, features2, hog_image2 = extract_hog(
    image2, "image2"
)


# Visualize HOG features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(gray_hog1, cmap="gray")
axes[0, 0].set_title("First Photograph")

axes[0, 1].imshow(hog_image1, cmap="gray")
axes[0, 1].set_title("HOG Features - Image 1")

axes[1, 0].imshow(gray_hog2, cmap="gray")
axes[1, 0].set_title("Second Photograph")

axes[1, 1].imshow(hog_image2, cmap="gray")
axes[1, 1].set_title("HOG Features - Image 2")

for ax in axes.flatten():
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "hog_visualization.png",
    dpi=150
)

plt.close()

print("\nHOG FEATURE EXTRACTION")
print("Image 1 HOG feature count:", len(features1))
print("Image 2 HOG feature count:", len(features2))


# --------------------------------------------------
# 4. SIFT KEYPOINT DETECTION
# --------------------------------------------------

if not hasattr(cv2, "SIFT_create"):
    raise RuntimeError(
        "SIFT is unavailable in this OpenCV installation. "
        "Check your OpenCV package."
    )

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(
    gray1, None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2, None
)

# Draw detected keypoints
keypoints_image1 = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoints_image2 = cv2.drawKeypoints(
    image2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints_image1.jpg"),
    keypoints_image1
)

cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints_image2.jpg"),
    keypoints_image2
)

print("\nSIFT KEYPOINT DETECTION")
print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))


# --------------------------------------------------
# 5. SIFT FEATURE MATCHING
# --------------------------------------------------

def match_sift_features(img1, img2, name):

    gray_img1 = cv2.cvtColor(
        img1, cv2.COLOR_BGR2GRAY
    )

    gray_img2 = cv2.cvtColor(
        img2, cv2.COLOR_BGR2GRAY
    )

    kp1, des1 = sift.detectAndCompute(
        gray_img1, None
    )

    kp2, des2 = sift.detectAndCompute(
        gray_img2, None
    )

    if des1 is None or des2 is None:
        print(name, ": Not enough features found.")
        return 0, 0, 0

    if len(des1) < 2 or len(des2) < 2:
        print(name, ": Not enough descriptors for matching.")
        return len(kp1), len(kp2), 0

    # Create Brute Force Matcher
    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    # Find two nearest matches
    matches = matcher.knnMatch(
        des1, des2, k=2
    )

    # Lowe's ratio test
    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Sort matches by distance
    good_matches = sorted(
        good_matches,
        key=lambda x: x.distance
    )

    # Draw feature matching result
    matching_image = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{name}.jpg"),
        matching_image
    )

    print(f"\n{name}")
    print("Keypoints in Image 1:", len(kp1))
    print("Keypoints in Image 2:", len(kp2))
    print("Good feature matches:", len(good_matches))

    return len(kp1), len(kp2), len(good_matches)


# --------------------------------------------------
# 6. ORIGINAL IMAGE MATCHING
# --------------------------------------------------

original_result = match_sift_features(
    image1,
    image2,
    "original_sift_matching"
)


# --------------------------------------------------
# 7. ROTATE ONE IMAGE AND MATCH AGAIN
# --------------------------------------------------

height, width = image2.shape[:2]

center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    30,
    1.0
)

rotated_image = cv2.warpAffine(
    image2,
    rotation_matrix,
    (width, height)
)

cv2.imwrite(
    str(OUTPUT_DIR / "rotated_image.jpg"),
    rotated_image
)

rotated_result = match_sift_features(
    image1,
    rotated_image,
    "rotated_sift_matching"
)


# --------------------------------------------------
# 8. RESIZE ONE IMAGE AND MATCH AGAIN
# --------------------------------------------------

resized_image = cv2.resize(
    image2,
    None,
    fx=1.25,
    fy=1.25,
    interpolation=cv2.INTER_CUBIC
)

cv2.imwrite(
    str(OUTPUT_DIR / "resized_image.jpg"),
    resized_image
)

resized_result = match_sift_features(
    image1,
    resized_image,
    "resized_sift_matching"
)


# --------------------------------------------------
# 9. COMPARISON RESULTS
# --------------------------------------------------

comparison = {
    "Original Images": original_result,
    "Rotated Image": rotated_result,
    "Resized Image": resized_result
}

report_path = OUTPUT_DIR / "matching_comparison.txt"

with open(report_path, "w") as file:

    file.write("SIFT FEATURE MATCHING COMPARISON\n\n")

    for name, result in comparison.items():

        kp1, kp2, good = result

        file.write(f"{name}\n")
        file.write(f"Keypoints in Image 1: {kp1}\n")
        file.write(f"Keypoints in Image 2: {kp2}\n")
        file.write(f"Good Matches: {good}\n\n")

        print(f"\n{name}")
        print("Good Matches:", good)


print("\nTASK 2 COMPLETED SUCCESSFULLY!")
print("All outputs saved inside:", OUTPUT_DIR)

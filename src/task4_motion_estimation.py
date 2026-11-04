
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "dataset" / "original_video.mp4"

OUTPUT_DIR = BASE_DIR / "outputs" / "task4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. OPEN VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Cannot open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if fps <= 0 or total_frames <= 0:
    cap.release()
    raise RuntimeError("Invalid video information.")

duration = total_frames / fps

print("VIDEO INFORMATION")
print("FPS:", round(fps, 2))
print("Total frames:", total_frames)
print("Duration:", round(duration, 2), "seconds")


# Select frames at 25%, 50% and 75% of the video
target_frames = {
    max(1, int(total_frames * 0.25)): "25",
    max(1, int(total_frames * 0.50)): "50",
    max(1, int(total_frames * 0.75)): "75"
}

print("\nSelected frame positions:", target_frames)


# --------------------------------------------------
# 3. INITIALIZE GMM BACKGROUND SUBTRACTOR
# --------------------------------------------------

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=32,
    detectShadows=True
)

kernel = np.ones((3, 3), np.uint8)


# --------------------------------------------------
# 4. INITIALIZE VARIABLES
# --------------------------------------------------

previous_gray = None

saved_results = {}

frame_index = 0


# --------------------------------------------------
# 5. PROCESS THE VIDEO FRAME BY FRAME
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Blur to reduce small image noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # --------------------------------------------------
    # A. GMM BACKGROUND SUBTRACTION
    # --------------------------------------------------

    raw_mask = background_subtractor.apply(frame)

    # Remove shadows (MOG2 represents shadows as gray values)
    _, gmm_mask = cv2.threshold(
        raw_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small noise
    gmm_mask = cv2.morphologyEx(
        gmm_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    gmm_mask = cv2.morphologyEx(
        gmm_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )


    # --------------------------------------------------
    # B. FRAME DIFFERENCING
    # --------------------------------------------------

    if previous_gray is not None:

        frame_difference = cv2.absdiff(
            previous_gray,
            gray
        )

        _, difference_mask = cv2.threshold(
            frame_difference,
            25,
            255,
            cv2.THRESH_BINARY
        )

        # Remove noise
        difference_mask = cv2.morphologyEx(
            difference_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Make moving regions more visible
        difference_mask = cv2.dilate(
            difference_mask,
            kernel,
            iterations=2
        )


    # --------------------------------------------------
    # C. LUCAS-KANADE OPTICAL FLOW
    # --------------------------------------------------

    optical_flow_image = frame.copy()

    tracked_points = 0
    average_movement = 0.0

    if previous_gray is not None:

        # Find important points in previous frame
        points_previous = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        if points_previous is not None:

            points_current, status, error = (
                cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    gray,
                    points_previous,
                    None,
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS
                        | cv2.TERM_CRITERIA_COUNT,
                        10,
                        0.03
                    )
                )
            )

            if points_current is not None and status is not None:

                good_new = points_current[
                    status.flatten() == 1
                ]

                good_old = points_previous[
                    status.flatten() == 1
                ]

                tracked_points = len(good_new)

                movement_values = []

                for new, old in zip(
                    good_new, good_old
                ):

                    x_new, y_new = new.ravel()
                    x_old, y_old = old.ravel()

                    # Calculate movement distance
                    distance = np.sqrt(
                        (x_new - x_old) ** 2
                        + (y_new - y_old) ** 2
                    )

                    movement_values.append(distance)

                    # Draw movement direction
                    cv2.arrowedLine(
                        optical_flow_image,
                        (int(x_old), int(y_old)),
                        (int(x_new), int(y_new)),
                        (0, 255, 0),
                        2,
                        tipLength=0.3
                    )

                    # Mark current point
                    cv2.circle(
                        optical_flow_image,
                        (int(x_new), int(y_new)),
                        4,
                        (0, 0, 255),
                        -1
                    )

                if movement_values:
                    average_movement = float(
                        np.mean(movement_values)
                    )


    # --------------------------------------------------
    # 6. SAVE SELECTED RESULTS
    # --------------------------------------------------

    if frame_index in target_frames:

        label = target_frames[frame_index]

        print(
            f"\nSaving results for {label}% frame..."
        )

        # Calculate moving pixel percentages
        diff_percentage = (
            np.count_nonzero(difference_mask)
            / difference_mask.size
        ) * 100

        gmm_percentage = (
            np.count_nonzero(gmm_mask)
            / gmm_mask.size
        ) * 100

        # Save original frame
        cv2.imwrite(
            str(OUTPUT_DIR / f"original_frame_{label}.jpg"),
            frame
        )

        # Save frame difference
        cv2.imwrite(
            str(OUTPUT_DIR / f"frame_difference_{label}.jpg"),
            difference_mask
        )

        # Save GMM foreground mask
        cv2.imwrite(
            str(OUTPUT_DIR / f"gmm_foreground_mask_{label}.jpg"),
            gmm_mask
        )

        # Save optical flow visualization
        cv2.imwrite(
            str(OUTPUT_DIR / f"optical_flow_{label}.jpg"),
            optical_flow_image
        )

        # Store information for comparison
        saved_results[label] = {
            "frame": frame.copy(),
            "difference": difference_mask.copy(),
            "gmm": gmm_mask.copy(),
            "flow": optical_flow_image.copy(),
            "diff_percentage": diff_percentage,
            "gmm_percentage": gmm_percentage,
            "tracked_points": tracked_points,
            "average_movement": average_movement
        }


    # Move to next frame
    previous_gray = gray.copy()

    frame_index += 1


# Release video
cap.release()


# --------------------------------------------------
# 7. CHECK RESULTS
# --------------------------------------------------

if not saved_results:
    raise RuntimeError(
        "No results were saved. Check the video file."
    )

print("\nVideo processing completed!")


# --------------------------------------------------
# 8. CREATE COMPARISON IMAGES
# --------------------------------------------------

for label, data in saved_results.items():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 11)
    )

    # Original frame
    axes[0, 0].imshow(
        cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB)
    )

    axes[0, 0].set_title("Original Video Frame")

    # Frame differencing
    axes[0, 1].imshow(
        data["difference"],
        cmap="gray"
    )

    axes[0, 1].set_title("Frame Differencing")

    # GMM
    axes[1, 0].imshow(
        data["gmm"],
        cmap="gray"
    )

    axes[1, 0].set_title("GMM Background Subtraction")

    # Lucas-Kanade
    axes[1, 1].imshow(
        cv2.cvtColor(data["flow"], cv2.COLOR_BGR2RGB)
    )

    axes[1, 1].set_title("Lucas-Kanade Optical Flow")

    for ax in axes.flatten():
        ax.axis("off")

    plt.suptitle(
        f"Motion Estimation Comparison - {label}% Video",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"motion_comparison_{label}.png",
        dpi=150
    )

    plt.close()


# --------------------------------------------------
# 9. GENERATE COMPARISON TABLE
# --------------------------------------------------

csv_path = OUTPUT_DIR / "motion_comparison.csv"

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Video Position",
        "Frame Difference Pixels (%)",
        "GMM Foreground Pixels (%)",
        "Optical Flow Tracked Points",
        "Average Movement (pixels)"
    ])

    for label, data in saved_results.items():

        writer.writerow([
            label + "%",
            round(data["diff_percentage"], 2),
            round(data["gmm_percentage"], 2),
            data["tracked_points"],
            round(data["average_movement"], 2)
        ])


# --------------------------------------------------
# 10. CREATE TEXT COMPARISON REPORT
# --------------------------------------------------

report_path = OUTPUT_DIR / "motion_comparison.txt"

with open(report_path, "w") as file:

    file.write("VIDEO MOTION ESTIMATION REPORT\n")
    file.write("==============================\n\n")

    file.write(
        f"Video duration: {duration:.2f} seconds\n"
    )

    file.write(f"Frame rate: {fps:.2f} FPS\n\n")

    for label, data in saved_results.items():

        file.write(f"\nVIDEO POSITION: {label}%\n")

        file.write(
            f"Frame Difference Moving Pixels: "
            f"{data['diff_percentage']:.2f}%\n"
        )

        file.write(
            f"GMM Foreground Pixels: "
            f"{data['gmm_percentage']:.2f}%\n"
        )

        file.write(
            f"Optical Flow Tracked Points: "
            f"{data['tracked_points']}\n"
        )

        file.write(
            f"Average Optical Flow Movement: "
            f"{data['average_movement']:.2f} pixels\n"
        )

    file.write("\nTECHNIQUE COMPARISON\n\n")

    file.write(
        "1. Frame Differencing compares consecutive frames "
        "to identify changes.\n"
    )

    file.write(
        "2. GMM Background Subtraction learns the background "
        "and identifies foreground objects.\n"
    )

    file.write(
        "3. Lucas-Kanade Optical Flow tracks selected feature "
        "points and estimates their movement.\n"
    )


# --------------------------------------------------
# 11. DISPLAY FINAL OUTPUT
# --------------------------------------------------

print("\nTASK 4 COMPLETED SUCCESSFULLY!")

print("Results saved inside:")
print(OUTPUT_DIR)

for label in saved_results:
    print(
        f"Comparison image: motion_comparison_{label}.png"
    )

print("\nComparison CSV saved:", csv_path)
print("Comparison report saved:", report_path)

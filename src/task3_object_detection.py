
import cv2
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PHOTO_PATH = BASE_DIR / "dataset" / "original_photo.jpeg"
SECOND_PHOTO_PATH = BASE_DIR / "dataset" / "second_photo.jpeg"
VIDEO_PATH = BASE_DIR / "dataset" / "original_video.mp4"

OUTPUT_DIR = BASE_DIR / "outputs" / "task3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD PHOTOGRAPHS
# --------------------------------------------------

photo = cv2.imread(str(PHOTO_PATH))
second_photo = cv2.imread(str(SECOND_PHOTO_PATH))

if photo is None:
    raise FileNotFoundError(f"Cannot find {PHOTO_PATH}")

if second_photo is None:
    raise FileNotFoundError(f"Cannot find {SECOND_PHOTO_PATH}")

print("Photographs loaded successfully!")



# --------------------------------------------------
# 3. IMPROVED HAAR CASCADE FACE DETECTION
# --------------------------------------------------

# Load frontal face classifier
frontal_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

frontal_cascade = cv2.CascadeClassifier(frontal_path)


# Load side-profile face classifier
profile_path = (
    cv2.data.haarcascades
    + "haarcascade_profileface.xml"
)

profile_cascade = cv2.CascadeClassifier(profile_path)


def detect_faces(image, filename):

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    # Improve image contrast
    gray = cv2.equalizeHist(gray)

    face_boxes = []

    # First try side-profile face detection
    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in profile_faces:
        face_boxes.append((x, y, w, h))

    # Check horizontally flipped image
    flipped = cv2.flip(gray, 1)

    flipped_faces = profile_cascade.detectMultiScale(
        flipped,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    image_width = gray.shape[1]

    for (x, y, w, h) in flipped_faces:

        # Convert coordinates back to original image
        original_x = image_width - (x + w)

        face_boxes.append(
            (original_x, y, w, h)
        )

    # If no profile face is found, try frontal detection
    if len(face_boxes) == 0:

        frontal_faces = frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(80, 80)
        )

        for (x, y, w, h) in frontal_faces:
            face_boxes.append((x, y, w, h))

    # Draw detected faces
    result = image.copy()

    for (x, y, w, h) in face_boxes:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            result,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imwrite(
        str(OUTPUT_DIR / filename),
        result
    )

    print(f"\nHaar Cascade - {filename}")
    print("Faces detected:", len(face_boxes))

    return len(face_boxes)


# Apply detection to both photographs
haar_count_1 = detect_faces(
    photo,
    "haar_face_original.jpg"
)

haar_count_2 = detect_faces(
    second_photo,
    "haar_face_second.jpg"
)

# --------------------------------------------------
# 4. LOAD PRETRAINED YOLO MODEL
# --------------------------------------------------

print("\nLoading pretrained YOLO model...")

model = YOLO("yolo26n.pt")

print("YOLO model loaded successfully!")


# --------------------------------------------------
# 5. YOLO OBJECT DETECTION
# --------------------------------------------------

def detect_objects(image, filename):

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes and class labels
    annotated_image = result.plot()

    cv2.imwrite(
        str(OUTPUT_DIR / filename),
        annotated_image
    )

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = model.names[class_id]

            detections.append(
                (class_name, confidence)
            )

    print(f"\nYOLO Detection - {filename}")
    print("Total objects detected:", len(detections))

    for name, confidence in detections:

        print(
            f"{name}: {confidence:.2f}"
        )

    return detections


# YOLO on original photographs
yolo_photo_1 = detect_objects(
    photo,
    "yolo_original_photo.jpg"
)

yolo_photo_2 = detect_objects(
    second_photo,
    "yolo_second_photo.jpg"
)


# --------------------------------------------------
# 6. EXTRACT SELECTED VIDEO FRAMES
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Cannot open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

if fps <= 0 or total_frames <= 0:
    cap.release()
    raise RuntimeError("Invalid video information.")

duration = total_frames / fps

print("\nVIDEO INFORMATION")
print("FPS:", fps)
print("Total frames:", int(total_frames))
print("Duration:", round(duration, 2), "seconds")


# Select 25%, 50% and 75% of the video
sample_times = [
    duration * 0.25,
    duration * 0.50,
    duration * 0.75
]

video_results = []

for index, seconds in enumerate(sample_times, start=1):

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        seconds * 1000
    )

    success, frame = cap.read()

    if not success:
        print("Could not extract frame:", index)
        continue

    # Save original frame
    frame_name = f"video_frame_{index}.jpg"

    cv2.imwrite(
        str(OUTPUT_DIR / frame_name),
        frame
    )

    # Apply YOLO detection
    detections = detect_objects(
        frame,
        f"yolo_video_frame_{index}.jpg"
    )

    video_results.append(
        (frame_name, detections)
    )

cap.release()


# --------------------------------------------------
# 7. SAVE COMPARISON REPORT
# --------------------------------------------------

report_path = OUTPUT_DIR / "detection_comparison.txt"

with open(report_path, "w") as file:

    file.write("OBJECT DETECTION COMPARISON\n")
    file.write("===========================\n\n")

    file.write("HAAR CASCADE RESULTS\n")

    file.write(
        f"Original photo faces: {haar_count_1}\n"
    )

    file.write(
        f"Second photo faces: {haar_count_2}\n\n"
    )

    file.write("YOLO PHOTO RESULTS\n")

    file.write(
        f"Original photo objects: {len(yolo_photo_1)}\n"
    )

    file.write(
        f"Second photo objects: {len(yolo_photo_2)}\n\n"
    )

    for name, detections in video_results:

        file.write(f"\n{name}\n")

        for class_name, confidence in detections:

            file.write(
                f"{class_name}: {confidence:.2f}\n"
            )

    file.write("\nCOMPARISON\n")

    file.write(
        "Haar Cascade detects frontal faces using a "
        "pretrained cascade classifier.\n"
    )

    file.write(
        "YOLO detects multiple supported object classes "
        "using a pretrained deep-learning model.\n"
    )


print("\nTASK 3 COMPLETED SUCCESSFULLY!")
print("All results saved inside:", OUTPUT_DIR)

# CCA 3 – Computer Vision Practical Assignment

An individual Python practical assignment covering **image restoration, image segmentation, feature extraction, object recognition, and video motion estimation** (Units 3, 4, and 5).

The project uses an original photograph, two photographs of the same person/scene for image matching, and an original self-recorded video. It implements the techniques listed in the assignment using OpenCV, NumPy, Matplotlib, scikit-image, and Ultralytics YOLO.

## 1. Assignment Objectives

The objective is to implement and compare important computer vision techniques across four practical tasks:

1. **Image Restoration and Segmentation:** Add noise to an image, restore it using filters, calculate image-quality metrics, and apply thresholding, morphological operations, edge detection, and watershed segmentation.
2. **Feature Extraction and Image Matching:** Extract HOG features, detect SIFT keypoints and descriptors, and match features between two photographs, including after rotation and resizing.
3. **Object Recognition:** Compare Haar Cascade (Viola–Jones) face detection with YOLO object detection on photographs and selected video frames.
4. **Video Motion Estimation:** Detect and estimate motion using frame differencing, GMM-based background subtraction, and Lucas–Kanade optical flow.

## 2. Technologies and Libraries

- **Python 3.11** – programming language
- **OpenCV** – image processing, video processing, Haar Cascade, SIFT, background subtraction, and optical flow
- **NumPy** – numerical operations and image arrays
- **Matplotlib** – visualization of processed images and comparison results
- **scikit-image** – HOG feature extraction and PSNR/SSIM metrics
- **Ultralytics YOLO** – pretrained object detection

The project was developed using a Python virtual environment. The OpenCV installation used for this project should include the Haar Cascade and SIFT APIs. The tested setup used `opencv-python==4.13.0.92`.

## 3. Project Folder Structure

Keep the input files and source files in the following structure. The filenames must match the paths used by the Python scripts.

```text
CCA3-Computer-Vision/
│
├── dataset/
│   ├── original_photo.jpeg
│   ├── second_photo.jpeg
│   ├── third_photo.jpeg
│   └── original_video.mp4
│
├── outputs/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
│
├── src/
│   ├── task1_image_processing.py
│   ├── task2_feature_matching.py
│   ├── task3_object_detection.py
│   └── task4_motion_estimation.py
│
├── .venv/
├── requirements.txt
└── README.md
```

### Input dataset

Place the original, unedited input files in the `dataset` folder:

| Filename | Description | Used in |
|---|---|---|
| `original_photo.jpeg` | Original photograph captured for the assignment | Task 1 and Task 3 |
| `second_photo.jpeg` | Second photograph of the same person or scene | Task 2 and Task 3 |
| `third_photo.jpeg` | Another photograph of the same person or scene | Task 2 |
| `original_video.mp4` | Original self-recorded video, approximately 15–30 seconds | Task 3 and Task 4 |

**Important:** The scripts use these exact filenames and extensions. If your files have different names or extensions, either rename them or update the corresponding paths at the beginning of the Python script. Preserve the original photograph and video without editing.

## 4. Environment Setup

### Requirements

- Windows 10/11 (the commands below use PowerShell)
- Python 3.11
- Visual Studio Code or another Python IDE
- Internet connection for installing packages and downloading the YOLO model weights on the first YOLO run

### Step 1 – Open the project folder

Open the `CCA3-Computer-Vision` folder in Visual Studio Code. Open the integrated terminal using **Terminal → New Terminal**.

### Step 2 – Create a virtual environment

Make sure Python 3.11 is selected. Check the installed Python versions with:

```powershell
py -0p
```

Create the virtual environment (run this only if `.venv` does not already exist):

```powershell
py -3.11 -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should show `(.venv)`. Confirm the Python version:

```powershell
python --version
```

It should report Python 3.11.x.

### Step 3 – Install dependencies

With the virtual environment activated, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main packages with:

```powershell
python -m pip install opencv-python==4.13.0.92 numpy matplotlib scikit-image ultralytics
```

Do not install multiple OpenCV wheel variants (for example, `opencv-python` and `opencv-contrib-python`) into the same environment unless you have a specific reason. They share the `cv2` namespace and can conflict.

### Step 4 – Verify the important libraries

Run:

```powershell
python -c "import cv2, numpy, matplotlib, skimage, ultralytics; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__); print('scikit-image:', skimage.__version__); print('Ultralytics:', ultralytics.__version__); print('Libraries imported successfully')"
```

You can also verify the Haar Cascade and SIFT APIs:

```powershell
python -c "import cv2; print('Haar Cascade API:', hasattr(cv2, 'CascadeClassifier')); print('SIFT API:', hasattr(cv2, 'SIFT_create'))"
```

Both should report `True` in the configured environment.

## 5. How to Execute the Assignment

Run the scripts from the **project root folder** (`CCA3-Computer-Vision`) with the virtual environment activated. Execute the tasks in order.

### Task 1 – Image Restoration and Segmentation

```powershell
python src/task1_image_processing.py
```

This script:

- Loads `dataset/original_photo.jpeg`.
- Adds Gaussian noise and salt-and-pepper noise.
- Applies median and Gaussian filtering to noisy images.
- Calculates PSNR and SSIM to compare processed images with the original.
- Converts the image to a binary image using Otsu thresholding.
- Applies erosion, dilation, opening, and closing.
- Performs Canny edge detection.
- Applies marker-based watershed segmentation.

**Expected output location:** `outputs/task1/`

The folder contains noisy images, restored images, binary and morphological results, Canny edges, watershed output, and a combined summary image (`task1_summary.png`). The script prints the PSNR and SSIM measurements in the terminal.

**Watershed note:** Watershed works best when the input contains multiple distinct or touching objects. Use a suitable object image/frame for this requirement and update the watershed input in the script if the original portrait does not produce clearly separated regions. Describe the actual result and any limitation in the report.

### Task 2 – Feature Extraction and Image Matching

```powershell
python src/task2_feature_matching.py
```

This script:

- Loads `dataset/second_photo.jpeg` and `dataset/third_photo.jpeg`.
- Extracts and visualizes Histogram of Oriented Gradients (HOG) features.
- Detects SIFT keypoints and computes descriptors.
- Matches SIFT features between the two photographs using a Brute Force matcher and Lowe's ratio test.
- Repeats matching after rotating and resizing one image.
- Saves the matching counts in a text comparison report.

**Expected output location:** `outputs/task2/`

Important outputs include `hog_visualization.png`, SIFT keypoint images, `original_sift_matching.jpg`, `rotated_sift_matching.jpg`, `resized_sift_matching.jpg`, and `matching_comparison.txt`.

### Task 3 – Object Recognition Using Haar Cascade and YOLO

```powershell
python src/task3_object_detection.py
```

This script:

- Uses Haar Cascade classifiers for frontal/profile face detection on the photographs.
- Uses the pretrained Ultralytics YOLO model for object detection on the photographs.
- Extracts three frames from the video and runs YOLO detection on those frames.
- Saves annotated images with bounding boxes, class labels, and YOLO confidence scores.
- Saves a text file summarizing the detections.

**Expected output location:** `outputs/task3/`

The first time the YOLO script runs, Ultralytics may download the pretrained model weights (configured in the script as `yolo26n.pt`). An internet connection is required for this initial download. Later runs can reuse the downloaded weights.

Keep examples of both correct detections and any false-positive or missed detections for the comparison in the report. Do not describe a detection as correct unless it has been checked against the actual image.

### Task 4 – Video Motion Estimation

```powershell
python src/task4_motion_estimation.py
```

This script:

- Reads the self-recorded video frame by frame.
- Applies frame differencing to consecutive frames.
- Applies MOG2 Gaussian Mixture Model (GMM) background subtraction.
- Uses Lucas–Kanade optical flow to track selected feature points between consecutive frames.
- Saves representative results at approximately 25%, 50%, and 75% of the video.
- Generates comparison images, a CSV file, and a text report.

**Expected output location:** `outputs/task4/`

Important outputs include original sample frames, frame-difference masks, GMM foreground masks, optical-flow visualizations, `motion_comparison_25.png`, `motion_comparison_50.png`, `motion_comparison_75.png`, `motion_comparison.csv`, and `motion_comparison.txt`.

Camera movement, lighting changes, shadows, and objects temporarily becoming part of the background can affect motion-estimation results. Mention relevant limitations when discussing the outputs.

## 6. Understanding the Output Metrics

- **PSNR (Peak Signal-to-Noise Ratio):** Measures the reconstruction quality of a processed image compared with the original. A higher PSNR generally indicates lower pixel-level error.
- **SSIM (Structural Similarity Index):** Compares structural similarity between the original and processed images. Values closer to 1 indicate greater similarity.
- **HOG feature vector:** A numerical representation of local gradient orientations and image shape information.
- **SIFT keypoints and matches:** Keypoints are distinctive image locations; matches indicate descriptor pairs that passed the matching test. A larger match count alone does not guarantee geometrically correct matches, so inspect the matching visualization.
- **YOLO confidence:** The model's confidence score for a predicted object class and bounding box.
- **Motion comparison measurements:** The frame-difference and GMM foreground percentages describe the fraction of pixels classified as changed/foreground in the selected frame. The optical-flow point count and average displacement describe the tracked points; they are not direct measurements of real-world speed.

## 7. Git Repository

Initialize and commit the project from the project root:

```powershell
git init
git add .
git commit -m "Complete CCA 3 computer vision assignment"
```

Create an empty repository on GitHub, add it as the remote, and push the project. For example (replace the URL with your own repository URL):

```powershell
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Do not commit the `.venv` folder or unnecessarily large raw video files. Add a `.gitignore` file containing at least:

```gitignore
.venv/
__pycache__/
*.py[cod]
```

Upload the original video to the Google Drive location provided by the instructor if it is too large for GitHub. Keep the required input files available to the person evaluating the project and explain in this README/report how to obtain them. Include the GitHub repository URL in the technical report.

## 8. Final Submission Checklist

- [ ] Original, unedited photograph included.
- [ ] Original, unedited self-recorded video included or shared through the instructor's Google Drive location.
- [ ] Second and third photographs included for feature matching.
- [ ] All four Python source files included.
- [ ] `requirements.txt` included.
- [ ] README and execution instructions included.
- [ ] Relevant output images, masks, visualizations, and comparison files included.
- [ ] Technical report includes explanations, captions, comparisons, limitations, and measurements.
- [ ] GitHub repository URL included in the technical report.
- [ ] 5–8 minute project explanation video recorded and uploaded to the instructor's Google Drive location.
- [ ] Required Google Drive/repository sharing permissions granted to the instructor's email addresses specified in the assignment.
- [ ] All scripts tested from the project root using the documented setup.

## 9. Notes

- Run each command from the project root, not from inside `src`.
- Keep the original input files unchanged.
- Output images and measurements are generated by the scripts under the corresponding `outputs/taskN/` folders.
- The YOLO model weights are downloaded separately by Ultralytics on the first run and are not part of the source code.
- The technical report should describe the results actually obtained from the selected input images and video, including any limitations or incorrect detections observed.

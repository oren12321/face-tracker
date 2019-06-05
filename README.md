# Face Tracker

This software automatically detects multiple faces from a video or a webcam, track those faces, and present them in separate stabilized windows.

The main idea of this software is face detection, tracking by optical flow and stabilization.

## How to use?

A precompiled version of the software can be used by running data/face-tracker.exe.
The software run according to the configuration parameters inside config/tracker-conf.ini. The configuration parameters are as follows:
* IS_CAMERA – If the value is 0 the program uses a given video file from the user, and if the value is 1 the program uses a webcam.
* RESOURCE – An identification number of the computer webcam (usually 0).
* VIDEO_PATH – The path of the given input video file.
* FPS – Frames per second if the video file in VIDEO_PATH.
* IS_RECORD – When equal to 0 the program not record the previewed results, and when equal to 1 the previewed results are been recorded.
* OUTPUT_VIDEO_PATH – The path of the output video in case that IS_RECORD=1 (The file name must be end with .avi extension).
* HAAR_FACE_FEATURES_PATH, HAAR_EYE_FEATURES_PATH, HAAR_NOSE_FEATURES_PATH, HAAR_MOUTH_FEATURES_PATH – Paths to the XML files that contain the features for the cascade detector. These files are located in the folder data/features.

An example for configuration file:
```ini
IS_CAMERA=1
RESOURCE=0
VIDEO_PATH=./videos/sample.mp4
FPS=29
IS_RECORD=0
OUTPUT_VIDEO_PATH=./results/sample_RES.avi
HAAR_FACE_FEATURES_PATH=./features/haarcascade_frontalface_alt2.xml
HAAR_EYE_FEATURES_PATH=./features/haarcascade_eye.xml
HAAR_NOSE_FEATURES_PATH=./features/nose.xml
HAAR_MOUTH_FEATURES_PATH=./features/mouth.xml
```

When the user runs the program, the first thing appears is a window with the selected video (or camera stream) with an ROI around each one of the detected faces.

In order to open a window that tracks one of the detected and stabilize it, the user need to double-click on one of the ROIs (The user can select multiple faces).

The user should press ESC in order to exit the program and stop the recording.

## The general algorithm

* Reading a frame from the video stream, converting it to grayscale and performing histogram equalization.
* If there is no initial face detection:
  * Detect faces with Haar cascade classifier.
  * Try to detect eyes nose and mouth of each detected face for higher precision.
  * Detect Shi-Tomasi features for each detected face.
  * Improve features by subpixel search.
* For the rest of the frames:
  * Bidirectional pyramidal optical flow by KLT for noise reduction.
  * Finding and performing geometrical transformation matrices for each ROI.
  * Finding inverse transformation matrices between the current ROIs to the first frame ROIs.
* In case that the user select video stabilization for one of the faces, a new window will be open and it will use the inverse transformation matrix.
* Perform the above steps on each frame until the end of the stream or until the user ends the program.

## Comments

The program is multithreaded and thus can support smoothly in multiple faces simultaneously.

The program in work online, which means that all the processing is done during the video sampling and the program will run less smooth if the user will open big number of windows.

A highly detailed explanation of the algorithm can be seen inside the source code.
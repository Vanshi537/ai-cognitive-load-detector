# AI Cognitive Load Detector using Facial Features

This project performs real-time facial feature extraction using MediaPipe and OpenCV to estimate cognitive load.

## Key Features
- Eye Aspect Ratio (EAR) for blink detection
- Gaze tracking & saccade detection
- Fixation detection logic
- Mouth openness & lip compression
- Brow height normalization
- Forehead wrinkle detection
- Baseline calibration system
- Head pose & posture

## How it Works
1. Captures webcam input
2. Extracts facial landmarks using MediaPipe
3. Computes multiple facial metrics
4. Tracks temporal changes (saccades, fixations)
5. Generates cognitive load indicators

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

## Run the Project
```bash
pip install -r requirements.txt
python main.py

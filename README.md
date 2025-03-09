# AI-Proctoring-System-with-Facial-Emotion-Recognition

Project Summary: AI Proctoring System with Facial Emotion Recognition

PROJECT OVERVIEW
An AI-powered proctoring system that uses facial recognition, emotion detection, eye tracking, and head pose estimation to monitor test-takers.
The system detects suspicious activities like multiple faces, looking away, and unusual head movements during an exam.

TECHNOLOGIES USED
OpenCV → Face detection, eye tracking, head pose estimation
Dlib → Facial landmarks detection
TensorFlow/Keras → Facial emotion recognition model
Pandas → Logging suspicious activities
NumPy → Data processing

KEY FUNCTIONALITIES
Face Detection 

Detects faces using OpenCV’s Haar cascade classifier.
Flags and logs if multiple faces are detected (possible cheating).

Emotion Recognition 

Loads a pre-trained deep learning model to predict emotions.
Uses 7-class emotion detection (angry, disgust, fear, happy, neutral, sad, surprise).

Eye Tracking 

Uses Dlib’s 68 facial landmarks to detect eye positions.
Logs if a student is looking away for an extended period.

Head Pose Estimation 🏷

Estimates head movement using solvePnP() in OpenCV.
Flags suspicious movements like excessive tilting or looking away.

Logging Suspicious Activities 

Saves timestamps and suspicious events in logs/suspicious_activities.csv.
Displays logged events in real-time on the terminal.

POSSIBLE ENHANCEMENTS
Blink detection to detect drowsiness or excessive eye closure.
Voice recording for additional cheating detection.
Improved head pose estimation using deep learning models instead of solvePnP().
Integrate with an exam portal for real-world usage.

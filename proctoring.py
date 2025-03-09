import cv2
import os
import pandas as pd
import dlib
import numpy as np
from datetime import datetime
from tensorflow.keras.models import model_from_json, Sequential

# Ensure logs folder exists
if not os.path.exists("logs"):
    os.makedirs("logs")

log_file = "logs/suspicious_activities.csv"

# Ensure log file exists with headers
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["Time", "Event"])
    df.to_csv(log_file, index=False)

def log_event(event):
    """Logs an event with timestamp into suspicious_activities.csv"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = pd.DataFrame([[timestamp, event]], columns=["Time", "Event"])
    data.to_csv(log_file, mode='a', header=False, index=False)
    print(f"Logged: {event}")

# Load emotion recognition model safely
try:
    with open("facialemotionmodel.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
    model.load_weights("facialemotionmodel.h5")
except Exception as e:
    print(f"Error loading emotion model: {e}")
    exit()

# Load face detection model
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Load Dlib model for eye tracking and head pose
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    print(f"Error: {predictor_path} not found. Please download the model.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Function to preprocess image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Head pose estimation reference points (6 points instead of 5)
object_pts = np.array([
    (0.0, 0.0, 0.0),       # Nose tip
    (-30.0, -30.0, -30.0),  # Left eye corner
    (30.0, -30.0, -30.0),   # Right eye corner
    (-45.0, 45.0, -30.0),   # Left mouth corner
    (45.0, 45.0, -30.0),    # Right mouth corner
    (0.0, -50.0, -30.0)     # Chin
], dtype=np.float32)


# Camera matrix approximation (Adjust based on camera resolution)
camera_matrix = np.array([[650, 0, 320], [0, 650, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Start webcam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret or frame is None:
        print("Warning: Failed to grab frame. Retrying...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 1:  # If multiple faces detected (Possible cheating)
        log_event("Multiple faces detected - Possible cheating")

    for face in faces:
        landmarks = predictor(gray, face)
        face_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y),  # Right mouth corner
            (landmarks.part(8).x, landmarks.part(8).y)  # Chin
        ], dtype=np.float32)

        # Estimate head pose
        _, rotation_vector, translation_vector = cv2.solvePnP(object_pts, face_points, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        if abs(angles[0]) > 15 or abs(angles[1]) > 15:  # If head moves too much
            log_event("Student looking away - Suspicious head movement")

        # Eye tracking
        left_eye_center = ((landmarks.part(37).x + landmarks.part(40).x) // 2,
                           (landmarks.part(37).y + landmarks.part(40).y) // 2)
        right_eye_center = ((landmarks.part(43).x + landmarks.part(46).x) // 2,
                            (landmarks.part(43).y + landmarks.part(46).y) // 2)

        # Draw head pose and eye tracking indicators
        cv2.putText(frame, f"Pitch: {angles[0]:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"Yaw: {angles[1]:.2f}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"Roll: {angles[2]:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)

        # Log if eyes look away
        if abs(left_eye_center[0] - right_eye_center[0]) > 35:  # Adjust threshold
            log_event("Student looking away - Eye movement detected")

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

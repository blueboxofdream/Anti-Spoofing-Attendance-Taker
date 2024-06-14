import cv2
import numpy as np
import subprocess
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('liveness_model.h5')

# Function to preprocess input image
def preprocess_image(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = resized_frame / 255.0  # Normalize
    return preprocessed_frame

# Function to perform real vs. spoof detection
def detect_real_vs_spoof(frame):
    preprocessed_frame = preprocess_image(frame)
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    return prediction

# Access the camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        prediction = detect_real_vs_spoof(face_roi)
        is_real_person = prediction > 0.5
        
        # Determine the color and text based on the prediction
        color = (0, 255, 0) if is_real_person else (0, 0, 255)  # Green for real, Red for spoof
        text = "Real Person" if is_real_person else "Spoofed Person"
        
        # Draw a rectangle around the face and add the text
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # If the person is real, display "p: Present"
        if is_real_person:
            cv2.putText(frame, "p:Present", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


    # cv2.putText(frame, "p:Present", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "q:Quit", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Real vs. Spoof Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.destroyAllWindows()  # Close the window
        break  # Break out of the loop to open attendance_taker.py

cap.release()
cv2.destroyAllWindows()

# Execute the attendance_taker.py script
subprocess.Popen(["python", "attendance_taker.py"])

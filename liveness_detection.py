import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import attendance_taker  # Importing the attendance taker module

# Load the face detection and liveness detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('liveness_model.h5')

# Define a function to preprocess the image for prediction
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize image to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

# Define a function to detect faces and liveness in a frame
def detect_faces_and_liveness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    show_button = False  # Initialize show_button variable
    
    # Initialize variables to track the presence of real and spoofed faces
    real_face_detected = False
    spoofed_face_detected = False
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        preprocessed_img = preprocess_image(face_img)
        prediction = model.predict(preprocessed_img)
        liveness_score = prediction[0][0]
        if liveness_score > 0.5:
            label = "Real Person"
            color = (0, 255, 0)  # Green color for real face
            real_face_detected = True
        else:
            label = "Spoofed Person"
            color = (0, 0, 255)  # Red color for spoofed face
            spoofed_face_detected = True
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Update show_button based on the presence of real and spoofed faces
    if real_face_detected and not spoofed_face_detected:
        show_button = True
    
    # Show button only if either all detected faces are real or there are no faces detected
    if not (real_face_detected and spoofed_face_detected):
        show_button = real_face_detected
    
    return frame, show_button

# Define the main GUI window
class LivenessDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Real-time Liveness Detection")

        self.video_label = tk.Label(master)
        self.video_label.pack()

        self.action_button = tk.Button(master, text="Give Your Attendance", command=self.perform_action)
        self.action_button.pack()

        self.quit_button = tk.Button(master, text="Quit", command=self.close)
        self.quit_button.pack()

        self.cap = cv2.VideoCapture(0)  # Open the default camera

        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret is None or frame is None:
            print("Error: No video input!!!")
            # You can display an error message to the user or attempt to recover gracefully
        else:
            frame, show_button = detect_faces_and_liveness(frame)
            
            # Convert BGR to RGB color space
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.config(image=img)
            
            if show_button:
                self.action_button.pack()  # Show button below the camera feed
            else:
                self.action_button.pack_forget()  # Hide button if no face detected
            
            self.video_label.after(10, self.update_video)  # Update every 10 milliseconds

    def perform_action(self):
        # Start a new thread to execute the attendance taker
        t = threading.Thread(target=self.execute_attendance_taker)
        t.start()

    def execute_attendance_taker(self):
        # Execute the attendance taker script
        attendance_taker.main()
        
        # Once the attendance is taken, update the GUI
        self.video_label.after(0, self.update_video)

    def close(self):
        self.cap.release()  # Release the camera
        self.master.destroy()

# Create the GUI window
root = tk.Tk()
app = LivenessDetectionApp(root)
root.mainloop()

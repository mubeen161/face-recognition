import face_recognition
import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Function to process the video feed and perform face recognition
def process_video_feed():
    # Load your training image
    image_path = "pro-pic.jpeg"  # Replace with the path to your image
    my_image = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(my_image)[0]

    # Create a list of known face encodings and their labels
    known_face_encodings = [my_face_encoding]
    known_face_names = ["Mubeen"]  # Replace with your name

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    # Streamlit video display container
    stframe = st.empty()

    stop_camera = False
    # Add a Stop button with a unique key (placed outside the loop)
    stop_button = st.button("Stop Camera", key="stop_button")

    while not stop_camera:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to access webcam. Please check your device.")
            break

        # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_frame = frame[:, :, ::-1]

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if the face matches the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Label the face with a name
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame back to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the video stream in Streamlit
        stframe.image(frame, channels="RGB", use_column_width=True)

        # Check if the Stop button is clicked
        if stop_button:
            stop_camera = True

    # Release the capture
    video_capture.release()

# Streamlit UI
st.title("Real-Time Face Recognition with Streamlit")
st.write("This application uses your webcam to perform face recognition.")

# Add a button to start the video stream with a unique key
if st.button("Start Camera", key="start_button"):
    process_video_feed()

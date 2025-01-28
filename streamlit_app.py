import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from geopy.geocoders import Nominatim  # To convert coordinates to location name
import pandas as pd  # To handle CSV file storage
from streamlit_js_eval import streamlit_js_eval  # For browser-based geolocation
import csv
from opencage.geocoder import OpenCageGeocode

# Your OpenCage API key
OPENCAGE_API_KEY = "e7b7c40cabc0426d83b537357b9e4780"  # Replace with your API key
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

# Define the CSV file to store location details
LOCATION_FILE = "user_locations.csv"
# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the model
classifier = load_model('model_78.h5')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi, verbose=0)[0]
                maxindex = int(np.argmax(prediction))
                output = emotion_labels[maxindex]
                label_position = (x, y - 10)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def get_user_location():
    """Fetch the user's location using JavaScript and save it with OpenCage."""
    location = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(
                position => resolve({latitude: position.coords.latitude, longitude: position.coords.longitude}),
                error => reject(error)
            );
        });
        """,
        key="get_location",
    )
    
    if location:
        lat = location.get("latitude")
        lon = location.get("longitude")
        if lat and lon:
            # Reverse geocode using OpenCage
            result = geocoder.reverse_geocode(lat, lon)
            if result and "formatted" in result[0]:
                location_name = result[0]["formatted"]
            else:
                location_name = "Unknown Location"

            # Save to CSV
            with open(LOCATION_FILE, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([lat, lon, location_name])

            st.success(f"Location detected: {location_name} (Latitude: {lat}, Longitude: {lon})")
            return lat, lon, location_name
        else:
            st.warning("Location could not be retrieved.")
            return None, None, None
    else:
        st.warning("Unable to fetch location. Please allow location access in your browser.")
        return None, None, None


def main():
    st.title("Real-Time Face Emotion Detection App with Location")
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Go to", ["Home", "Live Face Emotion Detection", "About"])

    if choice == "Home":
        st.write("""
            ## This app detects facial emotions in real-time using a pre-trained CNN model.
            This Web App is Made by Ahmed Islam, Tahir Mushtaq, Mubashar Nisar & Shams ul Hadi
            """
            )
        
         # Add an image on the home page
        st.image(
            "./Images/home image.png",  # image path
            caption="Real-Time Emotion Detection in Action",
            use_column_width=True
        )
        
        st.write("""
            ### Steps to Use:
            1. Navigate to **Live Face Emotion Detection** from the sidebar.
            2. Click **Start** to open your webcam and analyze emotions.
            3. Express different emotions and see real-time predictions.
        """)
       

    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed with Location Detection")
        st.subheader("Step 1: Detect your location")
    
        lat, lon, location_name = get_user_location()

        if lat and lon:
            st.map(data={"latitude": [lat], "longitude": [lon]})
            st.write(f"Location fetched successfully! Exact location: {location_name}")
            st.subheader("Step 2: Express your emotions!")
            webrtc_streamer(
                key="example",
                video_processor_factory=VideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
            )
        else:
            st.error("Unable to fetch location. Please allow location access and refresh the page.")

    elif choice == "About":
        st.subheader("About")
        st.write("""
            This app uses:
            - **Convolutional Neural Networks (CNN)** for emotion recognition.
            - **OpenCV** for face detection.
            - **Streamlit** and **streamlit-webrtc** for building an interactive real-time application.
            - **Browser Geolocation** for fetching user location.

            Built with love for demonstrating AI's capabilities in real-time.
        """)

if __name__ == "__main__":
    main()

import cv2
import numpy as np
from joblib import load
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
load_dotenv()
# ---------- Load .npz data ----------
def load_npz_data(path):
    data = np.load(path)
    return data['images'], data['labels']

# ---------- Emotion mappings ----------
emotion_to_genre = {
    0: 'sighu moose wala',       # Angry
    1: 'karan aujla',  # Disgust
    2: 'B praak',    # Fear
    3: 'haryanvi',      # Happy
    4: 'ammy virk',      # Neutral
    5: 'jhol',   # Sad
    6: 'punjabi'       # Surprise
}

EMOTIONS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# ---------- Spotify API setup ----------
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret")
))

# Global variable to store the model
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = load("emotion_classifier.joblib")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model file 'emotion_classifier.joblib' not found. "
                "Please run train_model.py first to train and save the model."
            )
    return _model

# ---------- Predict emotion and recommend a song ----------
def predict_emotion_and_song(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "No Face Detected", None

    (x, y, w, h) = faces[0]
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, (48, 48))
    # Flatten and normalize the image
    face_flat = face.reshape(1, -1).astype('float32') / 255.0

    model = get_model()  # Load model only when needed
    label = model.predict(face_flat)[0]
    mood = EMOTIONS[label]
    genre = emotion_to_genre[label]

    results = sp.search(q=f'genre:{genre} songs panjabi ', type='track', limit=1)
    if results['tracks']['items']:
        song = results['tracks']['items'][0]
        return mood, {
            "name": song["name"],
            "artist": song["artists"][0]["name"],
            "url": f"https://open.spotify.com/track/{song['id']}"
        }

    return mood, None

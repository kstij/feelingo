import cv2
import time
import streamlit as st
from deepface import DeepFace
import numpy as np
import spotipy
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from spotipy.oauth2 import SpotifyClientCredentials
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
finale = 'happy'
def detect_emotion():
    st.header("Real-time Face Emotion Detection")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Error: Could not open video capture device.")
        return

    video_capture.set(cv2.CAP_PROP_FPS, 5)

    start_time = time.time()  # get the start time
    emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0}

    image_placeholder = st.empty()

    while (time.time() - start_time) < 10:  # loop for 10 seconds
        ret, frame = video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = frame[y:y+h, x:x+w]
                emotion = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                dominant_emotion = emotion[0]['dominant_emotion']
                emotion_counts[dominant_emotion] += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
                image_placeholder.image(frame, channels="RGB", use_column_width=True)
            # Display the frame as an image using streamlit
            


        if cv2.waitKey(1) == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    maxs = 0
    finale = 'happy'

    # print the emotion counts
    for emotion, count in emotion_counts.items():
        if count >= maxs:
            finale = emotion
            maxs=count
            if finale == 'disgust':
                finale = 'sad'
            elif finale == 'neutral':
                finale = 'normal'
            elif finale == 'surprise':
                finale = 'excited'

    st.subheader("Overall Emotion Dominance : " + finale)

detect_emotion()

#Authentication - without user

def recommend_songs(emotion, num_to_select):
    num_songs=20
    # create a new array with the normalized features for the input emotion
    input_emotion = train.loc[train[f"emotion_{emotion}"] == 1].values[:, 0:]
    # find the nearest neighbors for the input emotion
    distances, indices = nn.kneighbors(input_emotion)
    # randomly select a subset of the top recommended songs
    top_songs = df1.iloc[indices[0]][['name', 'artist']].head(num_songs)
    selected_songs = top_songs.sample(n=num_to_select, replace=False)
    return selected_songs

client_credentials_manager = SpotifyClientCredentials(client_id="11a16a8384d4478b8c7bda02bc4ca803", client_secret="a1756e9d3498487e87f52f41d90c3741")
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
playlist_link = "https://open.spotify.com/playlist/78Sdax2ygDKQiVtbUtzjWQ"
playlist_URI = playlist_link.split("/")[-1].split("?")[0]
track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
# Retrieve the list of tracks in the playlist
results = sp.playlist_tracks(playlist_URI)
tracks = results['items']
while results['next']:
   results = sp.next(results)
   tracks.extend(results['items'])
features  = []
for i in range(100):
    features.append(sp.audio_features(track_uris)[i])
details = []
emotions = ["happy","sad","fear","normal","angry"]
for i,track in enumerate(tracks):
  track_name = track['track']['name']
  track_artist = track['track']['artists'][0]['name']
  track_album = track['track']['album']['name']
  track_popularity = track['track']['popularity']
  track_emotion = random.choice(emotions)
  track_acousticness = features[i]['acousticness']
  track_id = features[i]['id']
  track_energy = features[i]['energy']
  track_loudness = features[i]['loudness']
  track_liveness = features[i]['liveness']
  track_danceability = features[i]['danceability']
  track_speechiness = features[i]['speechiness']
  track_instrumentalness = features[i]['instrumentalness']
  track_tempo = features[i]['tempo']


  details.append({
    'name': track_name,
    'artist': track_artist,
    'album': track_album,
    'popularity': track_popularity,
    'emotion': track_emotion,
    'acousticness': track_acousticness,
    'id': track_id,
    'energy': track_energy,
    'loudness': track_loudness,
    'liveness': track_liveness,
    'danceability': track_danceability,
    'speechiness': track_speechiness,
    'instrumentalness': track_instrumentalness,
    'tempo': track_tempo

  })  
df1 = pd.DataFrame(details)
df1['album'] = df1['album'].str.split('(').str[0]
c = df1
df1['popularity'] = df1['popularity'].apply(lambda x: int(x/5))
emotions_encoded = pd.get_dummies(df1['emotion'], prefix='emotion')
emotions_df = df1
d = ['name','artist','album','id']
df1 = df1[d]
emotions_df = emotions_df.drop(['name','artist','album','id','popularity','emotion'], axis=1)
emotions_normalized = (emotions_df - emotions_df.mean()) / emotions_df.std()  
emotions_normalized = emotions_df - emotions_df.mean()
train = pd.concat([emotions_encoded, emotions_normalized], axis=1)
normalized = pd.concat([df1, emotions_encoded, emotions_normalized], axis=1)
nn = NearestNeighbors(n_neighbors=20, algorithm='ball_tree')
nn.fit(train)
output = recommend_songs(finale, 10)
output.to_string(index=False)

df = output
st.subheader("Your Playlist Suggestion")
# Display the DataFrame as a Streamlit table
st.table(df.style.set_properties(**{'background-color': 'white',
                                     'color': 'black',
                                     'font-size': '14px'}))


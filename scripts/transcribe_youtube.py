"""
a script to search for youtube videos, extract the first 10 seconds audio and transcribe it

install the following when you run it on google colab
!pip install --upgrade google-api-python-client
!pip install --upgrade google-auth-oauthlib google-auth-httplib2
!pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"
!pip install git+https://github.com/openai/whisper.git
!sudo apt update && sudo apt install ffmpeg
!pip install moviepy
"""

import os
from datetime import datetime, timedelta

import numpy as np
import torchaudio
import whisper
import youtube_dl
from googleapiclient.discovery import build
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# step 1: search for relevant youtube videos and  get the list of video ids
YOUTUBE_API_KEY = ""
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

SEARCH_PARAM = "AI"
request = youtube.search().list(
    part="snippet",
    q=SEARCH_PARAM,
    type="video",
    order="viewCount",
    maxResults=25,
    publishedAfter=datetime(2024, 1, 1).isoformat() + "Z",
    videoDuration="short",
    regionCode="US",
    relevanceLanguage="en",
)
response = request.execute()
for item in response["items"]:
    title = item["snippet"]["title"]
    if title.isascii():
        date = datetime.strptime(
            item["snippet"]["publishTime"], "%Y-%m-%dT%H:%M:%SZ"
        ).strftime("%Y-%m-%d")
        print(f"Published at: {date} | {title=}, Video ID: {item['id']['videoId']}")


video_ids = [
    item["id"]["videoId"]
    for item in response["items"]
    if item["snippet"]["title"].isascii()
]


# step 2: download the videos
ydl_opts = {"outtmpl": "./videos/%(title)s-%(id)s.%(ext)s"}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([f"https://www.youtube.com/watch?v={id}" for id in video_ids])


# step 3: extract the audio after clipping to the first 10 seconds
os.makedirs("audio", exist_ok=True)
MAX_DURATION = 10  # seconds
for file_name in os.listdir("./videos"):
    video = VideoFileClip(os.path.join("videos", file_name)).subclip(0, MAX_DURATION)
    audio = video.audio
    audio.write_audiofile(
        os.path.join("audio", "".join(file_name.split(".")[:-1]) + ".mp3")
    )

# step 4 load whisper base model and transcribe
model = whisper.load_model("base.en")
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

transcribed_result = {}
for file_name in tqdm(os.listdir("./audio")):
    print(f"transcribing {file_name=}")
    text = model.transcribe(os.path.join("audio", file_name))["text"]
    transcribed_result[file_name] = text

print(transcribed_result)

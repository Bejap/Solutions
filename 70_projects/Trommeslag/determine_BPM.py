from pydub import AudioSegment
import librosa
import numpy as np

this_file = "tambourine_90bpm.mp3"
sound = AudioSegment.from_mp3(this_file)
sound.export("output.wav", format="wav")

y, sr = librosa.load("output.wav")

onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

intervals = np.diff(onset_times)
bpm = 60 / np.mean(intervals)

differ_bpm = np.std(intervals)

print(f"bpm: {bpm} +/- {differ_bpm}")

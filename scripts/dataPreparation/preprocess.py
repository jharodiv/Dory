import os 
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

SAMPLERATE = 16000
DURATION = 2
N_MELS = 40
DATADIR ='../../data'
OUTDIR = '../../features'


classes = ['wake','not_wake','noise']

os.makedirs(OUTDIR, exist_ok=True)

def extractFeatures(file_path):
    y, sr = librosa.load(file_path, sr = SAMPLERATE)
    if len(y) < SAMPLERATE * DURATION:
        pad = SAMPLERATE * DURATION - len(y)
        y = np.pad(y, (0, pad))
    else:
        y = y[:SAMPLERATE * DURATION]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    meldb = librosa.power_to_db(mel, ref=np.max)
    return meldb

data = []
labels = []


for label in classes:
    classdir = os.path.join(DATADIR, label)
    for file in tqdm(os.listdir(classdir), desc=label):
        if file.endswith('.wav'):
            path = os.path.join(classdir, file)
            features = extractFeatures(path)
            data.append (features)
            labels.append(label)

np.save(os.path.join(OUTDIR, 'X.npy'), np.array(data))
np.save(os.path.join(OUTDIR, 'y.npy'), np.array(labels))
print("✅ Features and labels saved")
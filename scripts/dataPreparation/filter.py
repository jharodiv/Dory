import pandas as pd
import shutil
import os

esc_csv = 'meta/esc50.csv'
esc_audio_dir = 'audio'
noise_dest = 'data/noise'
os.makedirs(noise_dest, exist_ok=True)

# Categories you want to include as noise
include_labels = ['crowd', 'dog', 'rain', 'wind', 'engine', 'clapping', 'laughing']

df = pd.read_csv(esc_csv)

for _, row in df.iterrows():
    if row['category'] in include_labels:
        src_path = os.path.join(esc_audio_dir, row['filename'])
        dst_path = os.path.join(noise_dest, row['filename'])
        shutil.copy(src_path, dst_path)

print("✅ Noise files copied to:", noise_dest)

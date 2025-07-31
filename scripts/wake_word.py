import os
import ffmpeg

gtts_dir = 'data/wake/gtts'
real_dir = 'data/wake/real'
combined_dir = 'data/wake/combined'

os.makedirs(gtts_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)

def convert_to_wav(src_path, dest_path):
    ffmpeg.input(src_path).output(
        dest_path,
        format='wav',
        ac=1,    
        ar='16000'       
    ).overwrite_output().run()


for filename in os.listdir(gtts_dir):
    if filename.lower().endswith(('.mp3', '.wav')):
        src = os.path.join(gtts_dir, filename)
        dest = os.path.join(combined_dir, f"gtts_{filename.split('.')[0]}.wav")
        convert_to_wav(src, dest)

for filename in os.listdir(real_dir):
    if filename.lower().endswith(('.mp3', '.wav')):
        src = os.path.join(real_dir, filename)
        dest = os.path.join(combined_dir, f"gtts_{filename.split('.')[0]}.wav")
        convert_to_wav(src, dest)

print("Combined and converted files saved to:", combined_dir)
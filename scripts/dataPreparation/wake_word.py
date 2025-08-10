import os
import subprocess


#Convert .m4a to wav
wake_dir = 'data/wake'
converted_dir = 'data/wake_wav'
os.makedirs(converted_dir, exist_ok=True)

for filename in os.listdir(wake_dir):
    if filename.endswith('.m4a'):
        src = os.path.join(wake_dir, filename)
        dst = os.path.join(converted_dir, os.path.splitext(filename)[0] + '.wav')
        subprocess.run([r'C:\Users\USER\scoop\shims\ffmpeg.exe','-i', src, dst]) #change where your ffmpeg is 

import os
import librosa
import soundfile as sf
import numpy as np

#Combine the clear recorder to the background noises

# Paths
wake_dir = 'data/wake_wav'  
noise_dir = 'data/noise'  
augmented_dir = 'data/augmented'

os.makedirs(augmented_dir, exist_ok=True)

def add_noise(wake_path, noise_path, snr_db=10):
    wake_audio, sr = librosa.load(wake_path, sr=16000)
    noise_audio, _ = librosa.load(noise_path, sr=16000)

    if len(noise_audio) < len(wake_audio):
        repeats = int(np.ceil(len(wake_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)
    noise_audio = noise_audio[:len(wake_audio)]

    wake_power = np.mean(wake_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)

    snr = 10 ** (snr_db / 10)
    required_noise_power = wake_power / snr

    noise_audio = noise_audio * np.sqrt(required_noise_power / noise_power)

    noisy_audio = wake_audio + noise_audio

    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1:
        noisy_audio = noisy_audio / max_val

    return noisy_audio, sr

noise_files = os.listdir(noise_dir)
wake_files = os.listdir(wake_dir)

for wake_file in wake_files:
    wake_path = os.path.join(wake_dir, wake_file)

    noise_file = np.random.choice(noise_files)
    noise_path = os.path.join(noise_dir, noise_file)

    noisy_audio, sr = add_noise(wake_path, noise_path, snr_db=10)

    out_path = os.path.join(augmented_dir, f"noisy_{wake_file}")
    sf.write(out_path, noisy_audio, sr)

print("Done creating noisy wake word samples.")

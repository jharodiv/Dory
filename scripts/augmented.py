import os 
import random
from pydub import AudioSegment

clean_dir = 'data/wake'
noise_dir = 'data/noise'
output_dir = 'data/augmented'


os.makedirs(output_dir, exist_ok=True)


def mix_noise(clean_audio, noise_audio, snr_db):
    if len(noise_audio) < len(clean_audio):
        repeat_times = (len(clean_audio) // len (noise_audio)) + 1
        noise_audio = noise_audio * repeat_times


    noise_audio = noise_audio[:len(clean_audio)]

    noise_gain = clean_audio.dBFS - noise_audio.dBFS - snr_db
    noise_audio = noise_audio.apply_gain(noise_gain)

    mixed = clean_audio.overlay(noise_audio)
    return mixed

for filename in os.listdir(clean_dir):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        clean_path = os.path.join(clean_dir, filename)
        clean_audio = AudioSegment.from_file(clean_path).set_channels(1).set_frame_rate(16000)


        noise_file = random.choice(os.listdir(noise_dir))
        noisepath = os.path.join(noise_dir, noise_file)
        noise_audio = AudioSegment.from_file(noisepath).set_channels(1).set_fram_rate(16000)

        snr = random.randint(5, 20)
        noisy = mix_noise(clean_audio, noise_audio, snr)


        outputpath = os.path.join(output_dir, f"noisy_{snr}db_{filename}")
        noisy.export(outputpath, format = "wav")
        print(f"Saved: {outputpath}")

print("All files augmented the noise")
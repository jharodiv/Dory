import os 
import librosa
import numpy as np
from tqdm import tqdm

SAMPLERATE = 16000
DURATION = 2
N_MELS = 40
EXPECTED_WIDTH = 63
DATADIR ='../../data'
OUTDIR = '../../features'

classes = ['wake','not_wake','noise']
os.makedirs(OUTDIR, exist_ok=True)

def extractFeatures(y, sr, expected_width=EXPECTED_WIDTH):
    """Extract features from audio waveform - SINGLE SOURCE OF TRUTH"""
    # Pad/Trim to fixed duration
    target_length = SAMPLERATE * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    
    # Extract mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    meldb = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize
    meldb = (meldb - np.mean(meldb)) / (np.std(meldb) + 1e-6)
    
    # Pad/Trim to expected width
    if meldb.shape[1] < expected_width:
        pad_width = expected_width - meldb.shape[1]
        meldb = np.pad(meldb, pad_width=((0,0), (0,pad_width)), mode='constant')
    else:
        meldb = meldb[:, :expected_width]
    
    return meldb

def augment_audio(y, sr):
    """Apply augmentations with duration consistency"""
    # Time stretch - CORRECT: keyword argument
    rate = np.random.uniform(0.8, 1.2)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    
    # Pitch shift - CORRECT: keyword-only parameters
    n_steps = np.random.randint(-2, 3)
    y_shift = librosa.effects.pitch_shift(y_stretch, sr=sr, n_steps=n_steps)
    
    # Ensure consistent duration after augmentation
    target_length = SAMPLERATE * DURATION
    if len(y_shift) < target_length:
        y_shift = np.pad(y_shift, (0, target_length - len(y_shift)))
    else:
        y_shift = y_shift[:target_length]
    
    return y_shift

data = []
labels = []

for label in classes:
    classdir = os.path.join(DATADIR, label)
    for file in tqdm(os.listdir(classdir), desc=label):
        if file.endswith('.wav'):
            try:
                path = os.path.join(classdir, file)
                y, sr = librosa.load(path, sr=SAMPLERATE)
                
                # Original features - USING THE SAME FUNCTION
                features = extractFeatures(y, sr)
                data.append(features)
                labels.append(label)
                
                # Augment minority classes
                if label in ['wake','not_wake', 'noise']:
                    for _ in range(2):
                        try:
                            y_aug = augment_audio(y, sr)
                            # SAME FUNCTION - NO DUPLICATION!
                            features_aug = extractFeatures(y_aug, sr)
                            data.append(features_aug)
                            labels.append(label)
                        except Exception as aug_e:
                            print(f"Augmentation error for {file}: {aug_e}")
                            continue
                        
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

# Save results
np.save(os.path.join(OUTDIR, 'X.npy'), np.array(data))
np.save(os.path.join(OUTDIR, 'y.npy'), np.array(labels))
print("✅ Features and labels saved")

# Print summary
print(f"Total samples: {len(data)}")
print(f"Feature shape: {data[0].shape if data else 'No data'}")
print(f"Label counts: {dict(zip(*np.unique(labels, return_counts=True))) if labels else 'No labels'}")
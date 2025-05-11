# Import necessary libraries
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

# Set the input and output paths
input_path = 'pianos-by-jtwayne-7-174717.wav'
output_path = 'outputaudio.wav'

# Step 1: Load audio file
y, sr = librosa.load(input_path, sr=None)
print(f"Audio loaded: {input_path}, duration = {len(y)/sr:.2f}s, sampling rate = {sr}Hz")

# Step 2: Compute Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
print(S.shape)
print(S)

# Convert to dB scale (logarithmic)
S_dB = librosa.power_to_db(S, ref=np.max)
print(S_dB.shape)
print(S_dB)

# Display original Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Mel Spectrogram')
plt.tight_layout()
# plt.show()

# Step 3: Apply style transfer on linear scale
# Create style matrix that amplifies high frequencies more
style_matrix = np.ones_like(S)
# Convert frequency index to Mel scale
mel_freqs = librosa.mel_frequencies(n_mels=S.shape[0])
# Find index where Mel frequency is above 3kHz
cutoff_freq = 2000
cutoff_idx = np.where(mel_freqs >= cutoff_freq)[0][0]

# Ampify frequencies above cutoff
style_matrix[cutoff_idx:] *= 2  # Double gain in linear scale

# Apply style transformation
S_styled = S * style_matrix

# Convert back to dB for visualization
S_styled_dB = librosa.power_to_db(S_styled, ref=np.max)

# Display styled Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_styled_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Stylized Mel Spectrogram')
plt.tight_layout()
# plt.show()

# Step 4: Reconstruct audio from styled Mel spectrogram
# Increase number of iterations for better reconstruction
y_out = librosa.feature.inverse.mel_to_audio(S_styled, sr=sr, n_iter=100)

# Normalize the audio to ensure it's loud enough
y_out = librosa.util.normalize(y_out)

# Step 5: Save the transformed audio
sf.write(output_path, y_out, sr)
print(f"Transformed audio saved to: {os.path.abspath(output_path)}")
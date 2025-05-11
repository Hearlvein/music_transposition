import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

# === CONFIGURATION ===
input_path = "pianos-by-jtwayne-7-174717.wav"
output_path = 'output_transposed.wav'

# === ÉTAPE 1 : Chargement de l'audio ===
y, sr = librosa.load(input_path, sr=None)
print(f"Audio chargé : {input_path}, durée = {len(y)/sr:.2f}s, échantillonnage = {sr}Hz")

# === ÉTAPE 2 : Calcul du Mel spectrogramme ===
# Mel spectrogramme = version modifiée d'un spectrogramme qui utilise l’échelle de Mel 
# (plus proche de la façon dont l’oreille humaine perçoit les sons.)
# le Mel spectrogramme est une “image” de la musique, dans un format que les modèles
# peuvent comprendre tout en restant fidèle à la perception humaine.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
print(S.shape)
print(S)
S_dB = librosa.power_to_db(S, ref=np.max)
print(S_dB.shape)
print(S_dB)

# Affichage du spectrogramme original
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogramme original')
plt.tight_layout()
# plt.show()

# === ÉTAPE 3 : Application d'une "transformation de style" simple ===
# (par exemple : amplification des hautes fréquences)
style_matrix = np.ones_like(S_dB)
style_matrix[80:] *= 1.5  # Amplifie les fréquences au-dessus de ~3kHz
S_styled = S_dB * style_matrix

# Affichage du spectrogramme transformé
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_styled, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogramme transformé ("style transfer")')
plt.tight_layout()
# plt.show()

# === ÉTAPE 4 : Reconstruction de l'audio ===
S_linear = librosa.db_to_power(S_styled)
y_out = librosa.feature.inverse.mel_to_audio(S_linear, sr=sr, n_iter=32)
y_out = librosa.util.normalize(y_out)

# === ÉTAPE 5 : Sauvegarde du fichier transformé ===
sf.write(output_path, y_out, sr)
print(f"Audio transformé sauvegardé dans : {os.path.abspath(output_path)}")

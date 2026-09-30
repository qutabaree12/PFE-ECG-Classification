import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

# ==============================
# Mapping des sous-classes vers les classes AAMI
# ==============================
class_mapping = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

# ==============================
# Fonctions d'augmentation
# ==============================
def add_noise(signal, noise_level=0.02):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def time_warp(signal, warp_strength=0.02):
    indices = np.arange(len(signal))
    warped_indices = indices + np.random.uniform(-warp_strength, warp_strength, size=len(signal)) * len(signal)
    warped_indices = np.clip(warped_indices, 0, len(signal) - 1).astype(int)
    return signal[warped_indices]

def scale_signal(signal, scale_factor_range=(0.98, 1.02)):
    scale_factor = np.random.uniform(*scale_factor_range)
    return signal * scale_factor

def augment_signal(signal, max_augmentations=1):
    augmentations = random.sample([add_noise, time_warp, scale_signal], k=max_augmentations)
    for aug in augmentations:
        signal = aug(signal)
    return signal

# ==============================
# Chargement des données
# ==============================
processed_data_dir = r"C:\Users\esp\Desktop\PFE\pretraitement\data_proc"
segments_dir = os.path.join(processed_data_dir, "segments")
augmented_segments_dir = r"C:\Users\esp\Desktop\PFE\pretraitement\AUG"

os.makedirs(augmented_segments_dir, exist_ok=True)
target_classes = ['N', 'S', 'V', 'F', 'Q']

print("Process started...")

# Vérification de l'existence du dossier
if not os.path.exists(segments_dir):
    print(f"❌ Erreur : Le dossier des segments n'existe pas : {segments_dir}")
    exit()

X, y = [], []
for patient_folder in os.listdir(segments_dir):
    patient_path = os.path.join(segments_dir, patient_folder)
    
    for file_name in os.listdir(patient_path):
        if file_name.endswith(".npy"):
            segment_data = np.load(os.path.join(patient_path, file_name), allow_pickle=True).item()
            signal = segment_data["signal"]
            label = segment_data["label"]
            
            mapped_label = class_mapping.get(label, None)
            if mapped_label in target_classes:
                X.append(signal)
                y.append(mapped_label)

X = np.array(X)
y = np.array(y)

# ==============================
# Sous-échantillonnage de la classe majoritaire 'N'
# ==============================
class_counts_before = Counter(y)
major_class = 'N'
subsample_rate = class_counts_before[major_class] // 4  # Réduire fortement 'N'

indices_by_class = {cls: np.where(y == cls)[0] for cls in target_classes}

if major_class in indices_by_class:
    indices_by_class[major_class] = np.random.choice(indices_by_class[major_class], subsample_rate, replace=False)

selected_indices = np.concatenate([indices for indices in indices_by_class.values()])
np.random.shuffle(selected_indices)

X = X[selected_indices]
y = y[selected_indices]

print(f"✅ Sous-échantillonnage terminé : {Counter(y)}")

# ==============================
# Séparation des données en train/test (80% train, 20% test)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Création des dossiers
for cls in target_classes:
    os.makedirs(os.path.join(augmented_segments_dir, cls), exist_ok=True)

test_dir = os.path.join(augmented_segments_dir, "test")
os.makedirs(test_dir, exist_ok=True)

# ==============================
# Augmentation uniquement sur l'ensemble d'entraînement
# ==============================
median_class_size = int(np.median([sum(y_train == cls) for cls in target_classes]))

def compute_augmentation_factor(class_counts):
    return {cls: (median_class_size // count if count < median_class_size else 0) for cls, count in class_counts.items()}

train_class_counts = {cls: sum(y_train == cls) for cls in target_classes}
augmentation_factors = compute_augmentation_factor(train_class_counts)
augmented_class_counts = train_class_counts.copy()

# Sauvegarde des données d'entraînement avec augmentation
for i, (signal, label) in enumerate(zip(X_train, y_train)):
    class_dir = os.path.join(augmented_segments_dir, label)
    file_name = f"segment_{i}.npy"
    np.save(os.path.join(class_dir, file_name), {"signal": signal, "label": label})

    for j in range(augmentation_factors[label]):
        augmented_signal = augment_signal(signal)
        aug_file_name = f"segment_{i}_{j}.npy"
        np.save(os.path.join(class_dir, aug_file_name), {"signal": augmented_signal, "label": label})
        augmented_class_counts[label] += 1

print(" Augmentation terminée.")

# Sauvegarde des données de test sans augmentation
for i, (signal, label) in enumerate(zip(X_test, y_test)):
    np.save(os.path.join(test_dir, f"test_segment_{i}.npy"), {"signal": signal, "label": label})

# ==============================
# Vérification finale
# ==============================
print(" Nombre de fichiers générés après augmentation:")
for label in target_classes:
    folder_path = os.path.join(augmented_segments_dir, label)
    num_files = len(os.listdir(folder_path))
    print(f"Class {label}: {num_files} fichiers")

# ==============================
# Visualisation de la distribution finale
# ==============================
plt.figure(figsize=(8, 5))
plt.bar(augmented_class_counts.keys(), augmented_class_counts.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.xlabel("Classes")
plt.ylabel("Nombre de segments")
plt.title("Distribution des classes après sous-échantillonnage et augmentation")
plt.show()

print("\n✅ Processus terminé avec succès.")
print(f"✅ Les fichiers sont stockés dans : {augmented_segments_dir}")
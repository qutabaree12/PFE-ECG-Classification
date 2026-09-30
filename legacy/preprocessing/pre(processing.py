import os
import numpy as np
import wfdb
import pywt
from scipy.signal import savgol_filter
from collections import Counter
import random
from sklearn.model_selection import train_test_split

# ============================== #
# Prétraitement et fonctions
# ============================== #

def baseline_correction(signal, window_length=101, polyorder=3):
    return signal - savgol_filter(signal, window_length, polyorder)

def preprocess_ecg(signal):
    wavelet = "db4"
    level = 3
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs_thresholded = [coeffs[0]]
    for c in coeffs[1:]:
        threshold = np.median(np.abs(c)) * 1.5
        coeffs_thresholded.append(pywt.threshold(c, threshold, mode="soft"))
    return pywt.waverec(coeffs_thresholded, wavelet)

def min_max_scaling(signal, feature_range=(-1, 1)):
    min_val, max_val = np.min(signal), np.max(signal)
    scale_min, scale_max = feature_range
    if max_val - min_val == 0:
        return np.zeros_like(signal)
    return scale_min + (signal - min_val) * (scale_max - scale_min) / (max_val - min_val)

# ============================== #
# Segmentation
# ============================== #

def segment_ecg(signal, annotation, fs, segment_length):
    peaks = annotation.sample
    labels = annotation.symbol
    segments = []
    segment_labels = []

    for peak_index in peaks:
        start = peak_index - segment_length // 2
        end = peak_index + segment_length // 2

        if start < 0:
            padding = abs(start)
            segment = np.pad(signal[0:end], (padding, 0), mode='constant')
            label = labels[0]
        elif end > len(signal):
            padding = end - len(signal)
            segment = np.pad(signal[start:], (0, padding), mode='constant')
            label = labels[-1]
        else:
            segment = signal[start:end]
            label = labels[np.where(peaks == peak_index)[0][0]]

        segments.append(segment)
        segment_labels.append(label)

    return np.array(segments), np.array(segment_labels)

# ============================== #
# Sauvegarde segments
# ============================== #

def save_segments_by_patient(record_list, raw_data_dir, segments_dir, segment_length=360):
    os.makedirs(segments_dir, exist_ok=True)

    for record_name in record_list:
        print(f"\n🔹 Traitement du patient : {record_name}")
        patient_segment_dir = os.path.join(segments_dir, record_name)

        if os.path.exists(patient_segment_dir):
            print(f"✔️ Segments déjà existants pour {record_name}, on saute.")
            continue

        try:
            record = wfdb.rdrecord(os.path.join(raw_data_dir, record_name))
            annotation = wfdb.rdann(os.path.join(raw_data_dir, record_name), 'atr')
            signal = record.p_signal[:, 0]
            fs = record.fs

            # Prétraitement
            ecg_corrected = baseline_correction(signal)
            ecg_filtered = preprocess_ecg(ecg_corrected)
            ecg_normalized = min_max_scaling(ecg_filtered)

            # Segmentation
            segments, labels = segment_ecg(ecg_normalized, annotation, fs, segment_length)

            # Sauvegarde
            os.makedirs(patient_segment_dir, exist_ok=True)
            for i, (segment, label) in enumerate(zip(segments, labels)):
                np.save(os.path.join(patient_segment_dir, f"segment_{i+1}.npy"), segment)
                np.save(os.path.join(patient_segment_dir, f"label_{i+1}.npy"), label)

            print(f"✔️ {len(segments)} segments sauvegardés pour {record_name}")

        except Exception as e:
            print(f" Erreur pour {record_name} : {e}")

# ============================== #
# Chargement + Mapping AAMI
# ============================== #

def load_data(segments_dir, target_classes):
    segments = []
    labels = []
    patient_ids = []  # Pour tracer les patients associés si besoin

    label_mapping = {
        'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
        'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
        'V': 'V', 'E': 'V',
        'F': 'F',
        '/': 'Q', 'f': 'Q', 'Q': 'Q'
    }

    for root, dirs, files in os.walk(segments_dir):
        # Ne pas traiter le répertoire racine lui-même
        if root == segments_dir:
            continue
            
        patient_id = os.path.basename(root)
        
        for file in files:
            if file.startswith("segment_") and file.endswith(".npy"):
                try:
                    segment_path = os.path.join(root, file)
                    label_path = os.path.join(root, file.replace("segment_", "label_"))

                    segment = np.load(segment_path)
                    label = np.load(label_path)

                    mapped_label = label_mapping.get(str(label.item()), None)
                    if mapped_label in target_classes:
                        segments.append(segment)
                        labels.append(mapped_label)
                        patient_ids.append(patient_id)
                except Exception as e:
                    print(f"Erreur lors du chargement de {file} : {e}")
                    continue

    return np.array(segments), np.array(labels), np.array(patient_ids)

# ============================== #
# Fonctions d'augmentation
# ============================== #

def add_noise(segment, noise_level=0.05):
    noise = np.random.normal(0, noise_level * np.std(segment), size=segment.shape)
    return segment + noise

def shift_segment(segment, max_shift=0.1):
    shift_amount = int(random.uniform(-max_shift, max_shift) * len(segment))
    shifted = np.roll(segment, shift_amount)
    if shift_amount > 0:
        shifted[:shift_amount] = np.mean(segment)
    elif shift_amount < 0:
        shifted[shift_amount:] = np.mean(segment)
    return shifted

def stretch_compress(segment, max_scale=0.1):
    scale = random.uniform(1 - max_scale, 1 + max_scale)
    new_len = int(len(segment) * scale)
    stretched = np.interp(np.linspace(0, len(segment)-1, new_len), np.arange(len(segment)), segment)
    if len(stretched) < len(segment):
        return np.pad(stretched, (0, len(segment) - len(stretched)), constant_values=np.mean(segment))
    else:
        return stretched[:len(segment)]

# ============================== #
# Augmentation équilibrée à exactement 1000 par classe
# ============================== #

def augment_segments_with_exact_balance(segments, labels, target_per_class=10000):
    """
    Équilibre parfaitement les classes à exactement target_per_class échantillons:
    - Augmente les classes minoritaires jusqu'à target_per_class
    - Réduit les classes majoritaires à target_per_class
    """
    class_counts = Counter(labels)
    augmented_segments = []
    augmented_labels = []

    for label in np.unique(labels):
        class_segments = segments[labels == label]
        n_samples = len(class_segments)
        
        if n_samples < target_per_class:
            # Classe minoritaire: on ajoute tous les échantillons originaux
            augmented_segments.extend(class_segments)
            augmented_labels.extend([label] * n_samples)
            
            # Puis on génère des échantillons supplémentaires
            to_generate = target_per_class - n_samples
            for _ in range(to_generate):
                idx = np.random.randint(0, n_samples)
                seg = class_segments[idx].copy()
                # Appliquer 1 à 3 augmentations aléatoires
                for func in random.sample([add_noise, shift_segment, stretch_compress], k=random.randint(1, 3)):
                    seg = func(seg)
                augmented_segments.append(seg)
                augmented_labels.append(label)
                
        elif n_samples > target_per_class:
            # Classe majoritaire: on sélectionne aléatoirement target_per_class échantillons
            selected_indices = np.random.choice(range(n_samples), target_per_class, replace=False)
            selected_segments = class_segments[selected_indices]
            augmented_segments.extend(selected_segments)
            augmented_labels.extend([label] * target_per_class)
            
        else:  # n_samples == target_per_class
            # Déjà équilibré, on ajoute simplement tous les échantillons
            augmented_segments.extend(class_segments)
            augmented_labels.extend([label] * n_samples)

    return np.array(augmented_segments), np.array(augmented_labels)

# ============================== #
# Split Train/Test + Balancing
# ============================== #

def split_and_balance_data(segments, labels, test_size=0.2, random_state=42, target_per_class=50000):
    """
    Réalise un split train/test au niveau des segments puis équilibre la partie train à exactement target_per_class
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        segments, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"\n🔹 Split train/test réalisé:")
    print(f"  - Train: {X_train.shape[0]} segments")
    print(f"  - Test: {X_test.shape[0]} segments")
    
    # Distribution initiale
    print(f"\n Distribution des classes avant balancing:")
    print(f"  - Train: {Counter(y_train)}")
    print(f"  - Test: {Counter(y_test)}")
    
    # Appliquer le balancing uniquement sur les données d'entraînement
    X_train_balanced, y_train_balanced = augment_segments_with_exact_balance(X_train, y_train, target_per_class)
    
    print(f"\n Distribution des classes après balancing:")
    print(f"  - Train (balancé): {Counter(y_train_balanced)}")
    print(f"  - Test (non balancé): {Counter(y_test)}")
    
    return (X_train_balanced, y_train_balanced), (X_test, y_test)

# ============================== #
# Sauvegarde des datasets
# ============================== #

def save_datasets(X_train, y_train, X_test, y_test, output_dir):
    """
    Sauvegarde les datasets train et test
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde des données d'entraînement
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    
    # Sauvegarde des données de test
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    print(f"\n Datasets sauvegardés dans: {output_dir}")
    print(f"  - X_train.shape: {X_train.shape}")
    print(f"  - X_test.shape: {X_test.shape}")

# ============================== #
# Main
# ============================== #

if __name__ == "__main__":
    raw_data_dir = r"C:\Users\Thinkpad\Desktop\études\s6\PFE\code\données\raw"
    processed_data_dir = r"C:\Users\Thinkpad\Desktop\études\s6\PFE\code\données\processed"
    segments_dir = os.path.join(processed_data_dir, "segments")
    datasets_dir = os.path.join(processed_data_dir, "datasets")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    record_list = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                   '111', '112', '113', '114', '115', '116', '117', '118', '119', '200',
                   '201', '202', '203', '205', '207', '208', '209', '210', '212', '213',
                   '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                   '231', '232', '233', '234']

    segment_length = 360

    # Sauvegarde segments si pas déjà fait
    save_segments_by_patient(record_list, raw_data_dir, segments_dir, segment_length)

    # Chargement avec mapping AAMI
    target_classes = ['N', 'S', 'V', 'F', 'Q']
    all_segments, all_labels, _ = load_data(segments_dir, target_classes)

    print(f"\n Segments chargés : {all_segments.shape}")
    print(f"Répartition initiale : {Counter(all_labels)}")

    # Split train/test + balancing sur train seulement avec exact 1000 par classe
    (X_train, y_train), (X_test, y_test) = split_and_balance_data(
        all_segments, all_labels, test_size=0.2, random_state=42, target_per_class=50000
    )

    # Sauvegarde des datasets
    save_datasets(X_train, y_train, X_test, y_test, datasets_dir)
    
    print("\n Traitement terminé!")
    print(f"  - Dimensions finales X_train: {X_train.shape}")
    print(f"  - Dimensions finales X_test: {X_test.shape}")
    print(f"  - Répartition train (balancée): {Counter(y_train)}")
    print(f"  - Répartition test (non balancée): {Counter(y_test)}")
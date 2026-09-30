import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pywt
from scipy.signal import savgol_filter
from scipy.stats import kurtosis
import os

# ===============================
# Fonctions de prétraitement
# ===============================

def calculate_snr(original, filtered):
    noise = original - filtered
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_rmse(original, filtered):
    return np.sqrt(np.mean((original - filtered) ** 2))

def calculate_sdr(original, filtered):
    distortion = original - filtered
    sdr = 10 * np.log10(np.sum(original ** 2) / np.sum(distortion ** 2))
    return sdr

def baseline_correction(signal, window_length=101, polyorder=3):
    return signal - savgol_filter(signal, window_length, polyorder)

def min_max_scaling(signal, feature_range=(-1, 1)):
    min_val, max_val = np.min(signal), np.max(signal)
    scale_min, scale_max = feature_range
    if max_val - min_val == 0:
        return np.zeros_like(signal)
    return scale_min + (signal - min_val) * (scale_max - scale_min) / (max_val - min_val)

# ===============================
# Paramètres du projet
# ===============================

# Dossier contenant les enregistrements
data_folder = r"C:\Users\Thinkpad\Desktop\études\s6\PFE\code\data\brutes"

# Liste les fichiers dans le dossier en enlevant l'extension pour rdrecord
records_list = []
for filename in os.listdir(data_folder):
    if filename.endswith(".hea"):  # On récupère uniquement les headers
        record_name = os.path.splitext(filename)[0]  # On enlève l'extension
        records_list.append(record_name)

# ===============================
# Traitement de TOUS les enregistrements
# ===============================

for record_name in records_list:
    try:
        record_path = os.path.join(data_folder, record_name)

        # Chargement du signal avec wfdb (PAS d'extension dans rdrecord)
        record = wfdb.rdrecord(record_path)

        # Extraction d'une dérivation ECG (ici MLII par ex, donc canal 0)
        ecg_signal = record.p_signal[:, 0]

        # Correction de la ligne de base
        ecg_corrected = baseline_correction(ecg_signal)

        # Filtrage par DWT
        wavelet, niveaux = "db4", 3
        coeffs = pywt.wavedec(ecg_corrected, wavelet, level=niveaux)

        coeffs_seuilles = []
        for i, c in enumerate(coeffs):
            if i == 0:
                coeffs_seuilles.append(c)
            else:
                seuil = np.median(np.abs(c)) * 1.5
                coeffs_seuilles.append(pywt.threshold(c, seuil, mode="soft"))

        dwt_filtered_signal = pywt.waverec(coeffs_seuilles, wavelet)

        # Normalisation Min-Max [-1, 1]
        dwt_filtered_normalized = min_max_scaling(dwt_filtered_signal, feature_range=(-1, 1))

        # Calcul des métriques
        snr_dwt = calculate_snr(ecg_corrected, dwt_filtered_signal)
        rmse_dwt = calculate_rmse(ecg_corrected, dwt_filtered_signal)
        sdr_dwt = calculate_sdr(ecg_corrected, dwt_filtered_signal)
        kurt_before = kurtosis(ecg_corrected)
        kurt_after = kurtosis(dwt_filtered_signal)

        # Affichage console des métriques de CHAQUE enregistrement
        print(f"\n--- Métriques pour l'enregistrement {record_name} ---")
        print(f"SNR DWT : {snr_dwt:.2f} dB")
        print(f"RMSE DWT : {rmse_dwt:.4f}")
        print(f"SDR DWT : {sdr_dwt:.2f} dB")
        print(f"Kurtosis avant filtrage : {kurt_before:.4f}")
        print(f"Kurtosis après filtrage : {kurt_after:.4f}")
        print("-------------------------------------------")

        # Si c'est l'enregistrement 100, on trace les courbes !
        if record_name == "100":
            fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            # Signal brut
            axs[0].plot(ecg_signal, color='gray')
            axs[0].set_title(f"ECG Brut - Enregistrement {record_name}")
            axs[0].grid()

            # Signal filtré
            axs[1].plot(dwt_filtered_signal, color='green')
            axs[1].set_title(f"ECG après Filtrage DWT - Enregistrement {record_name}")
            axs[1].grid()

            # Signal normalisé
            axs[2].plot(dwt_filtered_normalized, color='blue')
            axs[2].set_title(f"ECG après Normalisation Min-Max [-1, 1] - Enregistrement {record_name}")
            axs[2].grid()

            plt.xlabel("Échantillons")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Erreur lors du traitement de l'enregistrement {record_name} : {e}")



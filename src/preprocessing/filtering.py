import numpy as np
import pywt


def preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    """
    Filtre le signal ECG avec une transformée en ondelettes discrète.

    La méthode utilisée dans le code historique du projet est :
    - ondelette : db4
    - niveau : 3
    - seuil : médiane des coefficients * 1.5
    - seuillage : soft

    Parameters
    ----------
    signal : np.ndarray
        Signal ECG d'entrée.

    Returns
    -------
    np.ndarray
        Signal ECG filtré.
    """
    wavelet = "db4"
    level = 3

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    coeffs_thresholded = [coeffs[0]]

    for coefficients in coeffs[1:]:
        threshold = np.median(np.abs(coefficients)) * 1.5
        coeffs_thresholded.append(
            pywt.threshold(
                coefficients,
                threshold,
                mode="soft",
            )
        )

    return pywt.waverec(coeffs_thresholded, wavelet)

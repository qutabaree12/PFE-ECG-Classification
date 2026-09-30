import numpy as np
from scipy.signal import savgol_filter


def baseline_correction(
    signal: np.ndarray,
    window_length: int = 101,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Corrige la ligne de base du signal ECG avec un filtre de Savitzky-Golay.

    Parameters
    ----------
    signal : np.ndarray
        Signal ECG d'entrée.
    window_length : int
        Longueur de la fenêtre du filtre.
    polyorder : int
        Ordre du polynôme utilisé par le filtre.

    Returns
    -------
    np.ndarray
        Signal ECG corrigé.
    """
    baseline = savgol_filter(signal, window_length, polyorder)
    return signal - baseline

import numpy as np


def min_max_scaling(
    signal: np.ndarray,
    feature_range: tuple[float, float] = (-1, 1),
) -> np.ndarray:
    """
    Normalise un signal ECG avec une mise à l'échelle Min-Max.

    Parameters
    ----------
    signal : np.ndarray
        Signal ECG d'entrée.
    feature_range : tuple[float, float]
        Intervalle cible de normalisation.

    Returns
    -------
    np.ndarray
        Signal normalisé.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)

    scale_min, scale_max = feature_range

    if max_val - min_val == 0:
        return np.zeros_like(signal)

    return scale_min + (
        (signal - min_val)
        * (scale_max - scale_min)
        / (max_val - min_val)
    )

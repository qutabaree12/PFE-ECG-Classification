import os
import numpy as np
import wfdb

from src.preprocessing.baseline_correction import baseline_correction
from src.preprocessing.filtering import preprocess_ecg
from src.preprocessing.normalization import min_max_scaling


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
            segment = np.pad(signal[0:end], (padding, 0), mode="constant")
            label = labels[0]

        elif end > len(signal):
            padding = end - len(signal)
            segment = np.pad(signal[start:], (0, padding), mode="constant")
            label = labels[-1]

        else:
            segment = signal[start:end]
            label = labels[np.where(peaks == peak_index)[0][0]]

        segments.append(segment)
        segment_labels.append(label)

    return np.array(segments), np.array(segment_labels)


def save_segments_by_patient(
    record_list,
    raw_data_dir,
    segments_dir,
    segment_length=360,
):
    os.makedirs(segments_dir, exist_ok=True)

    for record_name in record_list:
        print(f"\n🔹 Traitement du patient : {record_name}")

        patient_segment_dir = os.path.join(
            segments_dir,
            record_name,
        )

        if os.path.exists(patient_segment_dir):
            print(
                f"✔️ Segments déjà existants pour {record_name}, on saute."
            )
            continue

        try:
            record = wfdb.rdrecord(
                os.path.join(raw_data_dir, record_name)
            )

            annotation = wfdb.rdann(
                os.path.join(raw_data_dir, record_name),
                "atr",
            )

            signal = record.p_signal[:, 0]
            fs = record.fs

            ecg_corrected = baseline_correction(signal)
            ecg_filtered = preprocess_ecg(ecg_corrected)
            ecg_normalized = min_max_scaling(ecg_filtered)

            segments, labels = segment_ecg(
                ecg_normalized,
                annotation,
                fs,
                segment_length,
            )

            os.makedirs(patient_segment_dir, exist_ok=True)

            for i, (segment, label) in enumerate(
                zip(segments, labels)
            ):
                np.save(
                    os.path.join(
                        patient_segment_dir,
                        f"segment_{i + 1}.npy",
                    ),
                    segment,
                )

                np.save(
                    os.path.join(
                        patient_segment_dir,
                        f"label_{i + 1}.npy",
                    ),
                    label,
                )

            print(
                f"✔️ {len(segments)} segments sauvegardés pour {record_name}"
            )

        except Exception as exc:
            print(f"❌ Erreur pour {record_name} : {exc}")

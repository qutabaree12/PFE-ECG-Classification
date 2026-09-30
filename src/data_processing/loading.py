import os
import numpy as np


AAMI_LABEL_MAPPING = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    "V": "V",
    "E": "V",

    "F": "F",

    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


def load_data(segments_dir, target_classes):
    segments = []
    labels = []
    patient_ids = []

    for root, _, files in os.walk(segments_dir):

        if root == segments_dir:
            continue

        patient_id = os.path.basename(root)

        for file in files:

            if not (
                file.startswith("segment_")
                and file.endswith(".npy")
            ):
                continue

            try:
                segment_path = os.path.join(root, file)

                label_path = os.path.join(
                    root,
                    file.replace("segment_", "label_"),
                )

                segment = np.load(segment_path)
                label = np.load(label_path)

                mapped_label = AAMI_LABEL_MAPPING.get(
                    str(label.item()),
                    None,
                )

                if mapped_label in target_classes:
                    segments.append(segment)
                    labels.append(mapped_label)
                    patient_ids.append(patient_id)

            except Exception as exc:
                print(
                    f"Erreur lors du chargement de {file} : {exc}"
                )

    return (
        np.array(segments),
        np.array(labels),
        np.array(patient_ids),
    )


def load_data_by_patient(segments_dir, target_classes):
    patient_data = {}

    for patient_id in os.listdir(segments_dir):

        patient_dir = os.path.join(
            segments_dir,
            patient_id,
        )

        if not os.path.isdir(patient_dir):
            continue

        segments = []
        labels = []

        for file in os.listdir(patient_dir):

            if not (
                file.startswith("segment_")
                and file.endswith(".npy")
            ):
                continue

            try:
                segment_path = os.path.join(
                    patient_dir,
                    file,
                )

                label_path = os.path.join(
                    patient_dir,
                    file.replace(
                        "segment_",
                        "label_",
                    ),
                )

                segment = np.load(segment_path)
                label = np.load(label_path)

                mapped_label = AAMI_LABEL_MAPPING.get(
                    str(label.item()),
                    None,
                )

                if mapped_label in target_classes:
                    segments.append(segment)
                    labels.append(mapped_label)

            except (IOError, ValueError) as exc:
                print(
                    f"Erreur lors du chargement de {file} : {exc}"
                )

        if segments and labels:
            patient_data[patient_id] = {
                "segments": np.array(segments),
                "labels": np.array(labels),
            }

    return patient_data

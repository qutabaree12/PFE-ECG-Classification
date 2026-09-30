import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_processing.augmentation import (
    augment_segments_with_exact_balance,
)


def split_data_by_patient_and_class(
    patient_data,
    test_size=0.2,
    random_state=42,
):
    """
    Split des données par patient.

    Cette fonction provient de seg-ines.py et a été conservée
    comme implémentation historique du projet.
    """
    np.random.seed(random_state)

    train_patients = []
    test_patients = []

    patient_classes = {}

    for patient_id, data in patient_data.items():
        patient_classes[patient_id] = set(
            data["labels"]
        )

    all_classes = set(
        label
        for classes in patient_classes.values()
        for label in classes
    )

    class_counts = {
        cls: sum(
            cls in patient_classes[pid]
            for pid in patient_classes
        )
        for cls in all_classes
    }

    minor_classes = {
        cls
        for cls, count in class_counts.items()
        if count <= 1
    }

    for patient_id, classes in patient_classes.items():

        if classes & minor_classes:

            if np.random.rand() < test_size:
                test_patients.append(patient_id)
            else:
                train_patients.append(patient_id)

        else:

            if np.random.rand() < test_size:
                test_patients.append(patient_id)
            else:
                train_patients.append(patient_id)

    return train_patients, test_patients


def build_patient_split(
    patient_data,
    train_patients,
    test_patients,
):
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for patient_id in train_patients:
        X_train.extend(
            patient_data[patient_id]["segments"]
        )

        y_train.extend(
            patient_data[patient_id]["labels"]
        )

    for patient_id in test_patients:
        X_test.extend(
            patient_data[patient_id]["segments"]
        )

        y_test.extend(
            patient_data[patient_id]["labels"]
        )

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
    )


def split_and_balance_data(
    segments,
    labels,
    test_size=0.2,
    random_state=42,
    target_per_class=10000,
):
    """
    Implémentation historique par split de segments.

    Conservée pour reproduire les anciens résultats du projet.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        segments,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    X_train_balanced, y_train_balanced = (
        augment_segments_with_exact_balance(
            X_train,
            y_train,
            target_per_class,
        )
    )

    return (
        X_train_balanced,
        y_train_balanced,
    ), (
        X_test,
        y_test,
    )


def save_datasets(
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    np.save(
        os.path.join(
            output_dir,
            "X_train.npy",
        ),
        X_train,
    )

    np.save(
        os.path.join(
            output_dir,
            "y_train.npy",
        ),
        y_train,
    )

    np.save(
        os.path.join(
            output_dir,
            "X_test.npy",
        ),
        X_test,
    )

    np.save(
        os.path.join(
            output_dir,
            "y_test.npy",
        ),
        y_test,
    )

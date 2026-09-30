import random
import numpy as np


def add_noise(segment, noise_level=0.05):
    noise = np.random.normal(
        0,
        noise_level * np.std(segment),
        size=segment.shape,
    )

    return segment + noise


def shift_segment(segment, max_shift=0.1):
    shift_amount = int(
        random.uniform(
            -max_shift,
            max_shift,
        )
        * len(segment)
    )

    shifted = np.roll(
        segment,
        shift_amount,
    )

    if shift_amount > 0:
        shifted[:shift_amount] = np.mean(segment)

    elif shift_amount < 0:
        shifted[shift_amount:] = np.mean(segment)

    return shifted


def stretch_compress(segment, max_scale=0.1):
    scale = random.uniform(
        1 - max_scale,
        1 + max_scale,
    )

    new_len = int(
        len(segment) * scale
    )

    stretched = np.interp(
        np.linspace(
            0,
            len(segment) - 1,
            new_len,
        ),
        np.arange(len(segment)),
        segment,
    )

    if len(stretched) < len(segment):
        return np.pad(
            stretched,
            (
                0,
                len(segment) - len(stretched),
            ),
            constant_values=np.mean(segment),
        )

    return stretched[:len(segment)]


def augment_segments_capped(
    segments,
    labels,
    class_caps,
):
    segments = np.array(segments)
    labels = np.array(labels)

    augmented_segments = []
    augmented_labels = []

    for label in np.unique(labels):

        segs = segments[labels == label]

        current_count = len(segs)

        target_count = class_caps.get(
            label,
            current_count,
        )

        if current_count > target_count:

            indices = np.random.choice(
                current_count,
                target_count,
                replace=False,
            )

            segs = segs[indices]

            augmented_segments.extend(segs)

            augmented_labels.extend(
                [label] * target_count
            )

        elif current_count < target_count:

            augmented_segments.extend(segs)

            augmented_labels.extend(
                [label] * current_count
            )

            needed = target_count - current_count

            for _ in range(needed):

                base = segs[
                    np.random.randint(
                        0,
                        current_count,
                    )
                ].copy()

                aug_funcs = random.sample(
                    [
                        add_noise,
                        shift_segment,
                        stretch_compress,
                    ],
                    k=random.randint(1, 2),
                )

                for func in aug_funcs:
                    base = func(base)

                augmented_segments.append(base)
                augmented_labels.append(label)

        else:

            augmented_segments.extend(segs)

            augmented_labels.extend(
                [label] * current_count
            )

    return (
        np.array(augmented_segments),
        np.array(augmented_labels),
    )


def augment_segments_with_exact_balance(
    segments,
    labels,
    target_per_class=10000,
):
    augmented_segments = []
    augmented_labels = []

    for label in np.unique(labels):

        class_segments = segments[
            labels == label
        ]

        n_samples = len(class_segments)

        if n_samples < target_per_class:

            augmented_segments.extend(
                class_segments
            )

            augmented_labels.extend(
                [label] * n_samples
            )

            to_generate = (
                target_per_class
                - n_samples
            )

            for _ in range(to_generate):

                idx = np.random.randint(
                    0,
                    n_samples,
                )

                seg = class_segments[
                    idx
                ].copy()

                for func in random.sample(
                    [
                        add_noise,
                        shift_segment,
                        stretch_compress,
                    ],
                    k=random.randint(1, 3),
                ):
                    seg = func(seg)

                augmented_segments.append(seg)
                augmented_labels.append(label)

        elif n_samples > target_per_class:

            indices = np.random.choice(
                n_samples,
                target_per_class,
                replace=False,
            )

            augmented_segments.extend(
                class_segments[indices]
            )

            augmented_labels.extend(
                [label] * target_per_class
            )

        else:

            augmented_segments.extend(
                class_segments
            )

            augmented_labels.extend(
                [label] * n_samples
            )

    return (
        np.array(augmented_segments),
        np.array(augmented_labels),
    )

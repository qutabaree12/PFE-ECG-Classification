import matplotlib.pyplot as plt


def plot_training_history(history):
    history_data = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history_data.get("loss", []), label="Training")
    axes[0].plot(
        history_data.get("val_loss", []),
        label="Validation",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history_data.get("accuracy", []),
        label="Training",
    )
    axes[1].plot(
        history_data.get("val_accuracy", []),
        label="Validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    return fig, axes

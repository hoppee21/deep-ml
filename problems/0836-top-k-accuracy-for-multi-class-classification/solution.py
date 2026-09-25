import numpy as np

def top_k_accuracy(y_scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """
    Compute Top-K accuracy for multi-class classification.

    Args:
        y_scores: Predicted scores of shape (n_samples, n_classes)
        y_true: True class labels of shape (n_samples,)
        k: Number of top predictions to consider

    Returns:
        Top-K accuracy as a float rounded to 4 decimal places
    """
    top_k_preds = np.argsort(
        -y_scores,
        axis=1,
        kind="stable"
    )[:, :k]
    correct = np.any(top_k_preds == y_true[:, None], axis=1)

    return round(np.mean(correct), 4)

# metrics/utils.py

def count_true_positives_negatives(predictions, targets, positive_label):
    """
    Count true positives and true negatives.

    Parameters:
    - predictions (list): List of predicted labels.
    - targets (list): List of true labels.
    - positive_label: The label to consider as positive.

    Returns:
    - tuple: A tuple containing the number of true positives and true negatives.
    """
    true_positives = 0
    true_negatives = 0

    for pred, target in zip(predictions, targets):
        if pred == positive_label and target == positive_label:
            true_positives += 1
        elif pred != positive_label and target != positive_label:
            true_negatives += 1

    return true_positives, true_negatives

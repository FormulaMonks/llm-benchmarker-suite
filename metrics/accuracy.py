# metrics/accuracy.py

def calculate_accuracy(true_positives, true_negatives, total_samples):
    """
    Calculate the accuracy.

    Parameters:
    - true_positives (int): Number of true positives.
    - true_negatives (int): Number of true negatives.
    - total_samples (int): Total number of samples.

    Returns:
    - float: The accuracy.
    """
    accuracy = (true_positives + true_negatives) / total_samples
    return accuracy

# metrics/f1_score.py

def calculate_f1_score(true_positives, false_positives, false_negatives):
    """
    Calculate the F1 score.

    Parameters:
    - true_positives (int): Number of true positives.
    - false_positives (int): Number of false positives.
    - false_negatives (int): Number of false negatives.

    Returns:
    - float: The F1 score.
    """
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

from metrics.f1_score import calculate_f1_score
from metrics.accuracy import calculate_accuracy
from metrics.bleu_score import calculate_bleu_score
from metrics.utils import count_true_positives_negatives

# Example usage:
# Assume we have predictions and true labels as follows:
predictions = [1, 0, 1, 1, 0]
true_labels = [1, 0, 0, 1, 1]

# Calculate true positives and true negatives for a binary classification task
true_positives, true_negatives = count_true_positives_negatives(predictions, true_labels, positive_label=1)
print("True Positives:", true_positives)
print("True Negatives:", true_negatives)

# Calculate F1 score, accuracy, and BLEU score
f1_score = calculate_f1_score(true_positives, false_positives, false_negatives)
accuracy = calculate_accuracy(true_positives, true_negatives, len(predictions))
bleu_score = calculate_bleu_score(reference_sentences, predicted_sentence)

print("F1 Score:", f1_score)
print("Accuracy:", accuracy)
print("BLEU Score:", bleu_score)

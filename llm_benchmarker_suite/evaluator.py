import metrics

def evaluate(model_output, ground_truth):
  """
  Evaluate the model output against ground truth labels
  
  Args:
    model_output: dict containing model output
    ground_truth: dict containing ground truth labels
  
  Returns:
    metrics: dict containing evaluation metrics
  """

  # Calculate accuracy
  accuracy = metrics.calculate_accuracy(model_output["predictions"], ground_truth["labels"])

  # Get confusion matrix
  cm = metrics.get_confusion_matrix(model_output["predictions"], ground_truth["labels"])

  # Calculate precision, recall, F1
  precision = metrics.compute_precision(cm)
  recall = metrics.compute_recall(cm)
  f1 = metrics.compute_f_score(cm)

  # Return metrics
  calculated_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall, 
    "f1": f1
  }

  return calculated_metrics

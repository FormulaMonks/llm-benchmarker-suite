from evaluator import evaluate
import json

model_output = json.load(open('model_output.json')) 
ground_truth = json.load(open('ground_truth.json'))

metrics = evaluate(model_output, ground_truth)
print(metrics)

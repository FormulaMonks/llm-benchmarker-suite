from evaluator import evaluate
import json

model_output = json.load(open('llm_benchmarker_suite/model_output.json')) 
ground_truth = json.load(open('llm_benchmarker_suite/ground_truth.json'))

calculated_metrics = evaluate(model_output, ground_truth)
print(calculated_metrics)

from eval_levels.large_language_model import load_language_model, generate_completion
from eval_levels.gpt_completions_api import get_completion_score

# Example usage:
model_name = "gpt2"  # Replace with the desired Hugging Face language model
api_key = "your_gpt_completions_api_key"  # Replace with the actual API key

# Load the large language model from Hugging Face
language_model = load_language_model(model_name)

# Generate a random prompt and completion from the language model
prompt = "Once upon a time, there was a beautiful princess."
completion_text = generate_completion(language_model, prompt)

# Get the score from the GPT completions API by comparing the original text with the completion
original_text = prompt
completion_score = get_completion_score(api_key, prompt, original_text, completion_text)

print("Completion Score:", completion_score)

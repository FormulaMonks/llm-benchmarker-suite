from eval_levels import HuggingFaceLanguageModel, ProprietaryLanguageModel

# Example usage with Hugging Face model
huggingface_model = HuggingFaceLanguageModel("gpt2")
input_text = "Once upon a time, there was a beautiful princess"
completion = huggingface_model.generate_completions(input_text, max_length=100, num_return_sequences=1)
print("Hugging Face completion:", completion)

# Example usage with a proprietary model
proprietary_model = ProprietaryLanguageModel(model_url="https://api.example.com", api_key="your_api_key")
input_text = "Once upon a time, there was a beautiful princess"
completion = proprietary_model.generate_completions(input_text, custom_argument="value")
print("Proprietary model completion:", completion)

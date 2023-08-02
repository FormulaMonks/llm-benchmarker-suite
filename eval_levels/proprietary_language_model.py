from language_model_api import HuggingFaceLanguageModel, AnthropicLanguageModel, GPTLanguageModel

# Example usage with Hugging Face model
huggingface_model = HuggingFaceLanguageModel("gpt2")
input_text = "Once upon a time, there was a beautiful princess"
completion = huggingface_model.generate_completions(input_text, max_length=100, num_return_sequences=1)
print("Hugging Face completion:", completion)

# Example usage with Anthropics model
anthropic_model = AnthropicLanguageModel(api_key="your_anthropic_api_key")
input_text = "Once upon a time, there was a beautiful princess"
completions = anthropic_model.generate_completions(input_text, custom_argument="value")
for idx, completion in enumerate(completions):
    print(f"Anthropic completion {idx + 1}: {completion}")

# Example usage with GPT model
gpt_model = GPTLanguageModel(model_url="https://api.example.com/gpt", api_key="your_gpt_api_key")
input_text = "Once upon a time, there was a beautiful princess"
completions = gpt_model.generate_completions(input_text, custom_argument="value")
for idx, completion in enumerate(completions):
    print(f"GPT completion {idx + 1}: {completion}")

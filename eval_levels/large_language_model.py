# eval_levels/large_language_model.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_language_model(model_name):
    """
    Load the large language model from Hugging Face.

    Parameters:
    - model_name (str): The name of the Hugging Face language model.

    Returns:
    - GPT2LMHeadModel: The loaded language model.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model

def generate_completion(language_model, prompt, max_length=50):
    """
    Generate a completion from the language model given a prompt.

    Parameters:
    - language_model (GPT2LMHeadModel): The loaded language model.
    - prompt (str): The input prompt for generating the completion.
    - max_length (int): The maximum length of the completion.

    Returns:
    - str: The generated completion.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(language_model.config.architectures[0])
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = language_model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    generated_completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_completion

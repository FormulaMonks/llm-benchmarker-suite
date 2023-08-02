from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .generic_language_model import GenericLanguageModel

class HuggingFaceLanguageModel(GenericLanguageModel):
    def __init__(self, model_name):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_completions(self, input_text, **kwargs):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.model.generate(input_ids, **kwargs)
        completion = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return completion

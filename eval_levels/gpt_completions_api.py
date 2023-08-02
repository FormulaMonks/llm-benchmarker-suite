# eval_levels/gpt_completions_api.py
import requests

def get_completion_score(api_key, prompt, original_text, completion_text):
    """
    Get the score from the GPT completions API by comparing the original text and completion.

    Parameters:
    - api_key (str): The API key for accessing the GPT completions API.
    - prompt (str): The input prompt used to generate the completion.
    - original_text (str): The original text (before completion).
    - completion_text (str): The generated completion.

    Returns:
    - int: A score from 1 to 10 indicating the strength of the language model's completion.
    """
    # Implement the logic to call the GPT completions API and compare the completion with the original text
    # You need to use the API key to authenticate with the API
    # Replace the following code with the actual API call

    api_url = "https://api.example.com/gpt_completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"prompt": prompt, "original_text": original_text, "completion_text": completion_text}

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        score = response.json()["score"]
    else:
        raise Exception("Failed to get completion score from GPT completions API.")

    return score

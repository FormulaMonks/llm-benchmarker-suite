# metrics/bleu_score.py
import nltk

def calculate_bleu_score(reference_sentences, predicted_sentence):
    """
    Calculate the BLEU score.

    Parameters:
    - reference_sentences (list): List of reference sentences as strings.
    - predicted_sentence (str): The predicted sentence.

    Returns:
    - float: The BLEU score.
    """
    reference_sentences = [nltk.word_tokenize(sentence) for sentence in reference_sentences]
    predicted_sentence = nltk.word_tokenize(predicted_sentence)
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference_sentences, predicted_sentence)
    return bleu_score

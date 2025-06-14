import random
from ..utilities import get_french_synonyms

def synonym_replacement(text, n=1, seed=42):
    """
    Replace n words with their synonyms.

    Args:
        text (str): The input text.
        n (int): Number of words to replace with synonyms.
        seed (int): Random seed for reproducibility.
    """

    random.seed(seed)  # Set the random seed for reproducibility

    words = text.split()
    if len(words) <= 1:
        return text
    
    new_words = words.copy()
    random_word_indices = random.sample(range(len(words)), min(n, len(words)))
    
    for idx in random_word_indices:
        word = words[idx]
        synonyms = get_french_synonyms(word)
        if synonyms:
            new_words[idx] = random.choice(synonyms)  # Choose a random synonym from the synonym list
    
    return ' '.join(new_words)
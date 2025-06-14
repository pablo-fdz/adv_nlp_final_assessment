# Download WordNet data if not already available
import os
import nltk
from nltk.corpus import wordnet

# Check and download NLTK data at module import time (only once)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading Open Multilingual WordNet...")
    nltk.download('omw-1.4')

def get_french_synonyms(word):
    """Get French synonyms for a word using WordNet."""

    synonyms = []
    for synset in wordnet.synsets(word, lang='fra'):
        for lemma in synset.lemmas(lang='fra'):
            if lemma.name() != word:
                # Clean up multi-word synonyms and underscores
                cleaned = lemma.name().replace('_', ' ')
                synonyms.append(cleaned)
    return list(set(synonyms))
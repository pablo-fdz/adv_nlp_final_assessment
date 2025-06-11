from nltk.stem import PorterStemmer

def stem(text):
    """
    Preprocess text by stemming tokens using the Porter Stemmer algorithm.
    Should just input a string which has been previously pre-processed, which at least removes
    the punctuation.
    
    Returns:
        str: A string of stemmed tokens.
    """

    tokens = text.split()  # Split input text based on whitespaces
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)
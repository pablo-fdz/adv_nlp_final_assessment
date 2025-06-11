from nltk.stem import WordNetLemmatizer  # For lemmatizing

def lemmatize(text):
    
    """
    Preprocess text by applying lemmatizer.
    Should just input a string which has been previously pre-processed, which at least removes
    the punctuation.

    Args:
        text (str): A string of tokens to be lemmatized.

    Returns:
        list: A list of lemmatized tokens.
    """

    tokens = text.split()  # Split input text based on whitespaces
    lemmatizer = WordNetLemmatizer()  # Initiallize lemmatizer
    lemmatized_tokens = []  # Initialize empty list to store lemmatized text
    for word in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(word))

    return " ".join(lemmatized_tokens)
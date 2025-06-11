import re

def lower_nostop(text, rm_stopwords = False, stopword_set = None, output_type = 'list'):
    
    """
    Preprocess text by:
       - Converting to lowercase.
       - Removing punctuation.
       - Tokenizing.
       - Removing stopwords (optional).
    
    Returns:
        list/string: A list (or a string) of (concatenated) tokens lowercased, without punctuation and (if set) without stopwords.
    """
    
    # Convert to lowercase
    text_lower = text.lower()
    tokens = re.findall(r"\w+", text_lower)  # Match all alphanumeric characters
    
    # Remove stopwords if desired
    if rm_stopwords == True:
        tokens = [token for token in tokens if token not in stopword_set]
    
    if output_type == 'list':
        return tokens
    elif output_type == 'string':
        # Join tokens into a single string
        return " ".join(tokens)
    else:
        raise ValueError("output_type must be either 'list' or 'string'")
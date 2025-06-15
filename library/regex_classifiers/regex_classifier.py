# Create a simple classifier using regex
def regex_classifier(text, pattern, match_label=1):

    """
    Classify text using a regex pattern.

    Args:
        text (str): The text to classify.
        pattern (re.Pattern): The regex pattern to match.
        match_label (int): The label the pattern matches (1 for positive, 0 for negative).
    
    Returns:
        int: 1 if the text matches the pattern, 0 otherwise.
    """
    
    if match_label == 1:  # If the pattern matches a positive review
        if pattern.search(text):
            return 1  # Positive review
        else:
            return 0  # Negative review
    
    if match_label == 0:  # If the pattern matches a negative review
        if pattern.search(text):
            return 0  # Negative review
        else:
            return 1  # Positive review


def regex_classifier_ext(text, pattern_pos, pattern_neg):
    
    """
    Classify text using two regex patterns: one for positive and one for negative reviews.

    Args:
        text (str): The text to classify.
        pattern_pos (re.Pattern): The regex pattern for positive reviews.
        pattern_neg (re.Pattern): The regex pattern for negative reviews.

    Returns:
        int: 1 for positive review, 0 for negative review.
    """
    
    count_pos = len(pattern_pos.findall(text))
    count_neg = len(pattern_neg.findall(text))
    
    if count_pos > count_neg:
        return 1  # Positive review
    else:
        return 0  # Negative review
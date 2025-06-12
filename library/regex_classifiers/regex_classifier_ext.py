# Create a simple classifier using regex
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
import re
import polars as pl

def extract_vocabulary(df, text_column='text', min_word_length=2):
    """
    Extract a vocabulary list from text data in a Polars DataFrame.
    
    Args:
        df (pl.DataFrame): DataFrame containing text data
        text_column (str): Column name containing the text
        min_word_length (int): Minimum word length to include in vocabulary
        
    Returns:
        list: List of unique words found in the text data
    """
    # Combine all texts into a single string
    all_texts = df.select(pl.col(text_column)).to_series().str.join(" ")
    
    # Convert to lowercase
    text = all_texts[0].lower()
    
    # Use regex to extract only letter sequences (including French accented chars)
    # This pattern matches groups of letters including French accented characters
    pattern = r'[a-zàáâäæçèéêëîïôœùûüÿ]+'
    words = re.findall(pattern, text)
    
    # Filter by length and create unique set
    unique_words = list(set([word for word in words if len(word) >= min_word_length]))
    
    return unique_words
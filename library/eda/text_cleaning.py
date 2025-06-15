from nltk.corpus import stopwords
from typing import List
import spacy
import re
import polars as pl
import string

nlp = spacy.load("fr_core_news_sm")

def _basic_cleaner(text: str, lemmatize: bool = False) -> List[str]:
    text = re.sub(r'\b[A-Z]\.(_)\b', '', text) # instance like "X._" or "X." which indicate people (victims, witnesses, or plaintiffs)
    text = re.sub(r'\bA{2,}\b', '', text)  # AAA instances, which indicate companies or organizations
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    # REMOVE STPWORDS AND TOKENIZE
    words = text.split()
    stop_words = set(stopwords.words('french'))
    custom_stopwords = {'a', 'pr', 'u'} # remove words that appear as the most frequent word in ALL categories
    stop_words.update(custom_stopwords)
    words = [w for w in words if w not in stop_words]

    # LEMMATIZE
    if lemmatize:
        doc = nlp(" ".join(words))
        words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return words

def clean_df(df: pl.DataFrame, lemmatize: bool = False) -> pl.DataFrame:
    if lemmatize:
        df = df.with_columns(pl.col('text')
                             .map_elements(lambda x: _basic_cleaner(x, lemmatize), return_dtype=pl.List(pl.Utf8()))
                             .alias('clean_text_lem')
                             )

    else:
        df = df.with_columns(pl.col('text')
                             .map_elements(lambda x: _basic_cleaner(x, lemmatize), return_dtype=pl.List(pl.Utf8()))
                             .alias('clean_text_no_lem')
                             )

    return df
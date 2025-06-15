from collections import defaultdict
from typing import Tuple, List, Dict
import polars as pl
import re

def regex_classifier_ext(df, pred_column: str, text_column: str, topwords_dict: dict) -> pl.DataFrame:
    """
    Makes predictions based on the regex patterns. 
    Predicts dismissal or approval according to the count of top words from the respective dictionary.

    Args:
        df (pl.DataFrame): DataFrame containing the text data.
        pred_column (str): Name of the column to store predictions.
        text_column (str): Name of the column containing the text to analyze.
        topwords_dict (dict): Dictionary with tuples as keys (label, legal area) and lists of top words as values.
    """
    # create the prediction column
    if pred_column not in df.columns:
        df = df.with_columns(pl.lit(None).alias(pred_column))

    # make empty predictions list with the size of df
    predictions = df.select(pred_column).to_series().to_list() if pred_column in df.columns else [None] * df.height

    # create dictionary of top words without counts
    topwords_dict_wo_counts = {label_area: [word for word, _ in words] for label_area, words in topwords_dict.items()}

    # initialize regex pattern dictionary
    pattern_dict = {}

    # get set of legal areas to iterate over them
    areas = set(area for _, area in topwords_dict_wo_counts.keys())

    for area in areas:
        pred_series = []
        pos_words = topwords_dict_wo_counts.get((1, area), []) # approved
        neg_words = topwords_dict_wo_counts.get((0, area), []) # dismissed

        pattern_pos = re.compile(r'\b(?:' + '|'.join(map(re.escape, pos_words)) + r')\b', flags=re.IGNORECASE)
        pattern_neg = re.compile(r'\b(?:' + '|'.join(map(re.escape, neg_words)) + r')\b', flags=re.IGNORECASE)

        pattern_dict[area] = (pattern_pos, pattern_neg)

        pattern_pos, pattern_neg = pattern_dict.get(area)

        for i, row in enumerate(df.iter_rows(named=True)):
            if row['legal area'] == area:
                tokens = row[text_column]
                text = ' '.join(tokens) if isinstance(tokens, list) else str(tokens)
                pos_count = len(pattern_pos.findall(text))
                neg_count = len(pattern_neg.findall(text))
                predictions[i] = 1 if pos_count > neg_count else 0

    all_preds = pl.Series(name=pred_column, values=predictions)
    df = df.with_columns([all_preds])

    return df
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from collections import Counter
from typing import Union

def get_top_words(
    df: pd.DataFrame,
    text_column: str,
    class_column: str = 'class',
    area_column: str = 'legal_area',
    top_n: int = 20) -> dict:
    """
    Compute top N words for each (class, legal area) combination from a tokenized text column.

    Parameters:
    - df: pandas DataFrame containing the data
    - text_column: the name of the column with tokenized text (list of strings)
    - class_column: name of the column with class labels (default: 'class')
    - area_column: name of the column with legal area labels (default: 'legal_area')
    - top_n: how many top words to return per (class, area)

    Returns:
    - A dictionary of the form:
      {
        (class1, area1): [('word1', count), ('word2', count), ...],
        (class1, area2): [...],
        ...
      }
    """
    results = {}
    
    # Group by class and legal area
    grouped = df.group_by([class_column, area_column])
    
    for (cls, area), group in grouped:
        # Flatten list of tokens for this group
        all_tokens = [token for tokens in group[text_column] for token in tokens]
        word_counts = Counter(all_tokens)
        top_words = word_counts.most_common(top_n)
        results[(cls, area)] = top_words
        
    return results


def plot_top_words(
    top_words_dict: dict,
    max_cols: int = 3,
    figsize_per_plot: tuple = (5, 4),
    title_prefix: str = "Top words") -> None:

    """
    Visualizes the top words by (class, legal area) using bar plots.

    Parameters:
    - top_words_dict: output from get_top_words_by_class_and_area()
                      (a dict of {(class, area): [(word, count), ...]})
    - max_cols: maximum number of columns in the subplot grid
    - figsize_per_plot: tuple defining width and height per subplot
    - title_prefix: prefix for each subplot title
    """
    num_plots = len(top_words_dict)
    num_cols = min(max_cols, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols
    figsize = (figsize_per_plot[0] * num_cols, figsize_per_plot[1] * num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, ((cls, area), word_counts) in enumerate(top_words_dict.items()):
        words, counts = zip(*word_counts)
        sns.barplot(
            x=list(counts), y=list(words), ax=axes[i], orient='h', palette="viridis"
        )
        axes[i].set_title(f"{title_prefix}: {cls} | {area}")
        axes[i].set_xlabel("Count")
        axes[i].set_ylabel("Word")

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

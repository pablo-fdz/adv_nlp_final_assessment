import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import polars as pl
from nltk.corpus import stopwords


def plot_wordclouds(
    df: pl.DataFrame,
    category_col: str = "legal area",
    text_col: str = "text",
    label_col: str = "label",
    max_words: int = 100,
    figsize: tuple = (12, 3),
    additional_stopwords: set = None
):
    """
    Creates word clouds by label (e.g., dismissed/approved) for each category (e.g., legal area), removing common stopwords.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        category_col (str): Column to group by (e.g., 'legal area').
        text_col (str): Column containing text data.
        label_col (str): Binary label column (0 = negative, 1 = positive).
        max_words (int): Max words in each word cloud.
        figsize (tuple): Size of each subplot row.
        additional_stopwords (set): Extra stopwords to remove (in addition to default ones).
    """
    stopwords_set = set(STOPWORDS)
    try:
        stopwords_set.update(stopwords.words('french'))
        stopwords_set.update(stopwords.words('italian'))
    except LookupError:
        import nltk
        nltk.download('stopwords')
        stopwords_set.update(stopwords.words('french'))
        stopwords_set.update(stopwords.words('italian'))

    if additional_stopwords:
        stopwords_set |= additional_stopwords

    categories = df.select(category_col).unique().to_series().to_list()
    n_rows = len(categories)

    fig, axes = plt.subplots(n_rows, 2, figsize=(figsize[0], figsize[1] * n_rows))

    if n_rows == 1:
        axes = [axes]  # ensure axes is always iterable per row

    for i, cat in enumerate(categories):
        row_df = df.filter(pl.col(category_col) == cat)

        text_neg = " ".join(row_df.filter(pl.col(label_col) == 0)[text_col].to_list())
        text_pos = " ".join(row_df.filter(pl.col(label_col) == 1)[text_col].to_list())

        palette = color_palettes[i % len(color_palettes)]
        color_func = make_color_func(palette)

        wc_neg = WordCloud(width=800, height=400, background_color='white', max_words=max_words, color_func=color_func).generate(text_neg)
        wc_pos = WordCloud(width=800, height=400, background_color='white', max_words=max_words, color_func=color_func).generate(text_pos)

        axes[i][0].imshow(wc_neg, interpolation='bilinear')
        axes[i][0].axis('off')
        axes[i][0].set_title(f"{cat.upper()} - Dismissed (0)")

        axes[i][1].imshow(wc_pos, interpolation='bilinear')
        axes[i][1].axis('off')
        axes[i][1].set_title(f"{cat.upper()} - Approved (1)")

    plt.tight_layout()
    plt.show()

def make_color_func(colors):
    i = 0
    n = len(colors)
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        nonlocal i
        color = colors[i]
        i = (i + 1) % n
        return color
    return color_func

pinkish = ["#FFC0CB", "#FFB6C1", "#FF69B4", "#DB7093"]
orangeish = ["#FFA07A", "#FF8C00", "#FF7F50", "#FF6347"]
blueish = ["#ADD8E6", "#87CEEB", "#4682B4", "#1E90FF"]
greenish = ["#90EE90", "#32CD32", "#228B22", "#006400"]
redish = ["#FF6347", "#FF4500", "#DC143C", "#B22222"]
purpleish = ["#D8BFD8", "#DA70D6", "#BA55D3", "#800080"]

color_palettes = [pinkish, orangeish, blueish, greenish, redish, purpleish]

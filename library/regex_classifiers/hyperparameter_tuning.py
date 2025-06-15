from library.regex_classifiers.regex_classifier_ext import regex_classifier_ext
from library.eda.top_words import get_top_words
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.cm import get_cmap

def _calculate_accuracies(valid_df: pl.DataFrame, train_df: pl.DataFrame, top_ns: list):
    accs_no_lem = []
    accs_lem = []

    print('This takes some time...')

    for n in top_ns:
        # 1. Get top words for this top_n
        train_topwords_no_lem = get_top_words(train_df,
                                            text_column='clean_text_no_lem',
                                            class_column='label',
                                            area_column='legal area',
                                            top_n=n)

        train_topwords_lem = get_top_words(train_df,
                                        text_column='clean_text_lem',
                                        class_column='label',
                                        area_column='legal area',
                                        top_n=n)

        # 2. Predict with regex-based classifier
        valid_df = regex_classifier_ext(valid_df, 'regex_pred_no_lem', 'clean_text_no_lem', train_topwords_no_lem)
        valid_df = regex_classifier_ext(valid_df, 'regex_pred_lem', 'clean_text_lem', train_topwords_lem)

        # 3. Calculate accuracy
        acc_no_lem = accuracy_score(valid_df['label'], valid_df['regex_pred_no_lem'])
        acc_lem = accuracy_score(valid_df['label'], valid_df['regex_pred_lem'])

        accs_no_lem.append(round(acc_no_lem, 4))
        accs_lem.append(round(acc_lem, 4))

    print('Accuracies calculated.')

    return accs_no_lem, accs_lem

def plot_accuracies_curve(valid_df: pl.DataFrame, train_df: pl.DataFrame, top_ns: list):
    """
    Plots the accuracies for different top_n values.
    
    Args:
        valid_df (pl.DataFrame): Validation DataFrame.
        train_df (pl.DataFrame): DataFrame containing the training data.
        top_ns (list): List of top_n values used for the predictions.
    """
    accs_no_lem, accs_lem = _calculate_accuracies(valid_df, train_df, top_ns)

    viridis = get_cmap('viridis')
    color_no_lem = viridis(0.2)
    color_lem = viridis(0.8)

    plt.figure(figsize=(6, 4))
    plt.plot(top_ns, accs_no_lem, marker='o', label='No Lemmatization', color=color_no_lem)
    plt.plot(top_ns, accs_lem, marker='s', label='With Lemmatization', color=color_lem)

    plt.xlabel('Number of Top Words used (Top N)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Top N for Rule-Based Classifier')
    plt.xticks(top_ns)
    plt.ylim(0.35, 0.80)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
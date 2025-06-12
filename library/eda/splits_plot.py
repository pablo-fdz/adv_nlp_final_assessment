import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import polars as pl
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import polars as pl
from itertools import product


def plot_splits(train_filtered: pl.DataFrame, 
                                     valid_filtered: pl.DataFrame, 
                                     test_filtered: pl.DataFrame,
                                     split_colors: dict = None,
                                     figsize=(12, 6),
                                     group_col: str = "legal area") -> None:
    """
    Plots a stacked bar chart of label proportions per category (e.g., legal area) for train/valid/test splits.

    Args:
        train_filtered (pl.DataFrame): Filtered training set.
        valid_filtered (pl.DataFrame): Filtered validation set.
        test_filtered (pl.DataFrame): Filtered test set.
        split_colors (dict): Optional dict of split -> CSS4 color names.
        figsize (tuple): Size of the plot.
        group_col (str): Column to group by on x-axis (e.g., 'legal area').

    Returns:
        None (displays the plot).
    """

    # Default colors if none provided
    if split_colors is None:
        split_colors = {
            "train": mcolors.CSS4_COLORS["lightsteelblue"],
            "valid": mcolors.CSS4_COLORS["cornflowerblue"],
            "test": mcolors.CSS4_COLORS["royalblue"],
        }

    # Add split column
    train_df2 = train_filtered.with_columns(pl.lit("train").alias("split"))
    valid_df2 = valid_filtered.with_columns(pl.lit("valid").alias("split"))
    test_df2 = test_filtered.with_columns(pl.lit("test").alias("split"))

    # Combine
    all_df = pl.concat([train_df2, valid_df2, test_df2])

    # Group and count
    counts = (
        all_df.group_by(["split", group_col, "label"])
        .len()
        .rename({"len": "count"})
    )

    # Normalize to proportions
    total_per_split = counts.group_by("split").agg(pl.sum("count").alias("total"))
    counts = counts.join(total_per_split, on="split")
    counts = counts.with_columns((pl.col("count") / pl.col("total")).alias("proportion"))

    # Convert to pandas
    df_plot = counts.to_pandas()
    df_plot["group"] = df_plot[group_col].astype(str)

    # Ensure all combinations exist
    all_groups = df_plot["group"].unique()
    all_splits = df_plot["split"].unique()
    all_labels = df_plot["label"].unique()

    full_index = pd.DataFrame(
        list(product(all_splits, all_groups, all_labels)),
        columns=["split", "group", "label"]
    )

    df_plot = pd.merge(full_index, df_plot, on=["split", "group", "label"], how="left")
    df_plot["proportion"] = df_plot["proportion"].fillna(0)

    # Pivot to wide format
    pivoted = df_plot.pivot_table(
        index=["group", "split"],
        columns="label",
        values="proportion",
        fill_value=0
    ).reset_index()

    # Plotting
    def shade_color(color_name, factor):
        c = mcolors.to_rgb(color_name)
        return tuple(min(1, x * factor) for x in c)

    splits = ["train", "valid", "test"]
    groups = pivoted['group'].unique()
    labels = [col for col in pivoted.columns if col not in ['split', 'group']]
    x = np.arange(len(groups))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    for i, split in enumerate(splits):
        split_data = pivoted[pivoted['split'] == split]
        split_data = split_data.set_index('group').reindex(groups).fillna(0)
        base_color = split_colors[split]
        bottom = np.zeros(len(groups))

        for j, label in enumerate(labels):
            heights = split_data[label].values
            shade_factor = 0.7 + 0.4 * (j / max(len(labels)-1, 1))
            color = shade_color(base_color, shade_factor)

            ax.bar(x + i * bar_width, heights, bar_width, bottom=bottom,
                   color=color, label=f"{split} - Label {label}")
            bottom += heights

    # Final plot styling
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_xlabel(group_col.title())
    ax.set_ylabel('Proportion')
    ax.set_title(f'Label Proportions by {group_col.title()} and Split')

    # De-duplicate legend
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
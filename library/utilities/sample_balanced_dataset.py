import polars as pl
from datasets import Dataset

# Function to sample balanced dataset
def sample_balanced_dataset(
        dataset: pl.DataFrame, 
        num_samples, 
        seed, 
        sample_proportion=0.5
    ):

    """
    Sample a balanced dataset with equal numbers of positive and negative examples.
    
    Args:
        dataset (pl.DataFrame): The input dataset containing 'text' and 'label' columns.
        num_samples (int): Total number of samples to return, must be even.
        seed (int): Random seed for reproducibility.
        sample_proportion (float): Proportion of positive samples to include in the balanced dataset.
    Returns:
        Dataset: A balanced Dataset object with equal numbers of positive and negative examples.
    """

    # Get positive and negative examples
    pos_examples = dataset.filter(pl.col('label') == 1)
    neg_examples = dataset.filter(pl.col('label') == 0)
    
    # Calculate the inverse of the sample proportion
    denominator = 1 / sample_proportion

    # Sample equal numbers from each class
    samples_per_class = num_samples // denominator
    
    if seed is not None:
        pos_sampled = pos_examples.sample(n=samples_per_class, shuffle=True, seed=seed)
        neg_sampled = neg_examples.sample(n=samples_per_class, shuffle=True, seed=seed)
    else:
        raise ValueError("Seed must be provided for reproducibility.")
    
    # Concatenate the sampled DataFrames, only the text and the label columns
    pos_sampled = pos_sampled.select(['text', 'label'])
    neg_sampled = neg_sampled.select(['text', 'label'])
    df = pl.concat([pos_sampled, neg_sampled], how='vertical')

    # Combine the datasets into a single Dataset object
    combined = Dataset.from_polars(df)
    
    return combined
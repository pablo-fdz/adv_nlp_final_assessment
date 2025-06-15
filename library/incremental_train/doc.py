import os
import polars as pl
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ..data_augmentation import augment_dataset

def tokenize(batch, tokenizer, max_length):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

def train_with_percentage(train_df, valid_df, percentage, model_name, max_length, num_labels, seed=42,
                          sample_proportion=0.5, augmentation_rate: float=None, augmentation_techniques: list=None):
    """
    Train the model with a specific percentage of the training data.

    Args:
        train_df (pl.DataFrame): The training dataset containing 'text' and 'labels'.
        valid_df (pl.DataFrame): The validation dataset containing 'text' and 'labels'.
        percentage (int): Percentage of the training data to use for training.
        model_name (str): Name of the pre-trained model to use.
        max_length (int): Maximum length for tokenization.
        num_labels (int): Number of labels for classification.
        seed (int): Random seed for reproducibility.
        sample_proportion (float): Proportion of positive samples to include in the balanced dataset.
        augmentation_rate (float): Rate of augmentations to apply per each augmentation technique,
            expressed as a proportion of the dataset size (e.g., 0.2 for 20%).
            This determines how many augmented examples will be generated.
        augmentation_techniques (list): List of initialized augmentation objects to apply to the training data.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_samples = int(len(train_df) * percentage / 100)
    
    # Ensure n_samples is at least 2 and even (for binary balance)
    n_samples = max(2, n_samples - n_samples % 2)

    # Calculate the inverse of the sample proportion
    denominator = 1 / sample_proportion

    # Sample equal numbers from each class
    samples_per_class = n_samples // denominator
    
    # Make a copy to avoid modifying the original DataFrame
    train_df_copy = train_df.clone()
    
    # Rename columns if needed
    if 'labels' in train_df_copy.columns:
        train_df_copy = train_df_copy.rename({'labels': 'label'})
    
    # Split by class
    pos_examples = train_df_copy.filter(pl.col('label') == 1)
    neg_examples = train_df_copy.filter(pl.col('label') == 0)
    
    # Check if we need to use replacement sampling
    pos_with_replacement = (samples_per_class > pos_examples.shape[0])
    neg_with_replacement = (samples_per_class > neg_examples.shape[0])
    
    # Sample from each class, using replacement if necessary
    if pos_with_replacement:
        print(f"Using sampling with replacement for positive class (need {samples_per_class}, have {pos_examples.shape[0]})")
    pos_sampled = pos_examples.sample(n=samples_per_class, shuffle=True, seed=seed, with_replacement=pos_with_replacement)
    
    if neg_with_replacement:
        print(f"Using sampling with replacement for negative class (need {samples_per_class}, have {neg_examples.shape[0]})")
    neg_sampled = neg_examples.sample(n=samples_per_class, shuffle=True, seed=seed, with_replacement=neg_with_replacement)
    
    # Combine the samples
    train_subset = pl.concat([pos_sampled, neg_sampled])
    train_subset = Dataset.from_polars(train_subset)

    # Augment dataset with specified techniques if provided
    if augmentation_rate is not None and augmentation_techniques is not None:
        train_subset = augment_dataset(
            dataset=train_subset,
            techniques=augmentation_techniques,
            augmentation_rate=augmentation_rate,
            seed=seed  # Ensure reproducibility across iterations
        )
    
    # Rename if needed for HF compatibility
    if 'label' in train_subset.column_names:
        train_subset = train_subset.rename_column('label', 'labels')
    
    val_set = Dataset.from_polars(valid_df.select(['text', 'labels']))
    train_subset = train_subset.map(lambda x: tokenize(x, tokenizer, max_length), batched=True, remove_columns=["text"])
    val_set = val_set.map(lambda x: tokenize(x, tokenizer, max_length), batched=True, remove_columns=["text"])
    train_subset.set_format(type='torch')
    val_set.set_format(type='torch')
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    san_model_name = model_name.split(sep='/')[-1]
    use_fp16 = torch.cuda.is_available()
    train_args = TrainingArguments(
        output_dir=os.path.join('models', 'part_3', 'a', f'cls_fine_tuning_{san_model_name}_{percentage}pct'),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        seed=seed,
        report_to="none",
        fp16=use_fp16,
        gradient_accumulation_steps=2
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_subset,
        eval_dataset=val_set.shuffle(seed=seed),
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
            "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1)),
            "precision": precision_score(p.label_ids, np.argmax(p.predictions, axis=1)),
            "recall": recall_score(p.label_ids, np.argmax(p.predictions, axis=1))
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    train_output = trainer.train()
    log_history = trainer.state.log_history
    epoch_logs = [log for log in log_history if 'epoch' in log]
    results_df = pl.DataFrame(epoch_logs)
    results_path = os.path.join('results', 'part_3', 'a', f'cls_fine_tuning_results_{san_model_name}_{percentage}pct.parquet')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.write_parquet(results_path)
    best_epoch = results_df.filter(pl.col('eval_loss') == results_df['eval_loss'].min())
    return {
        'percentage': percentage,
        'best_epoch': best_epoch,
        'results_path': results_path
    }

def run_incremental_training(train_df, valid_df, model_name, max_length, num_labels, seed=42,
                             sample_proportion=0.5, augmentation_rate: float=None, augmentation_techniques: list=None):
    """
    Run training with different percentages of the dataset (1%, 10%, 25%, 50%, 75%, 100%).

    Args:
        train_df (pl.DataFrame): The training dataset containing 'text' and 'labels'.
        valid_df (pl.DataFrame): The validation dataset containing 'text' and 'labels'.
        model_name (str): Name of the pre-trained model to use.
        max_length (int): Maximum length for tokenization.
        num_labels (int): Number of labels for classification.
        seed (int): Random seed for reproducibility.
        sample_proportion (float): Proportion of positive samples to include in the balanced dataset.
        augmentation_rate (float): Rate of augmentations to apply per each augmentation technique,
            expressed as a proportion of the dataset size (e.g., 0.2 for 20%).
            This determines how many augmented examples will be generated.
        augmentation_techniques (list): List of initialized augmentation objects to apply to the training data.
    """
    percentages = [1, 10, 25, 50, 75, 100]
    all_results = []
    for pct in percentages:
        print(f"\nTraining with {pct}% of the data...")
        result = train_with_percentage(train_df, valid_df, pct, model_name, max_length, num_labels, seed,
                                       sample_proportion, augmentation_rate, augmentation_techniques)
        all_results.append(result)
        print(f"\nBest metrics for {pct}% of data:")
        print(result['best_epoch'])
    summary_data = []
    for result in all_results:
        best_epoch = result['best_epoch']
        summary_data.append({
            'percentage': result['percentage'],
            'eval_loss': best_epoch['eval_loss'].item(),
            'eval_accuracy': best_epoch['eval_accuracy'].item(),
            'eval_f1': best_epoch['eval_f1'].item(),
            'eval_precision': best_epoch['eval_precision'].item(),
            'eval_recall': best_epoch['eval_recall'].item()
        })
    summary_df = pl.DataFrame(summary_data)
    print("\nSummary of results across all percentages:")
    print(summary_df)
    summary_path = os.path.join('results', 'part_3', f"cls_fine_tuning_summary_{model_name.split('/')[-1]}.parquet")
    summary_df.write_parquet(summary_path)
    print(f'\nSummary saved to: {summary_path}')
    return summary_df 
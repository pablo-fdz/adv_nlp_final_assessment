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
from library.utilities import sample_balanced_dataset

def tokenize(batch, tokenizer, max_length):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

def train_with_percentage(train_df, valid_df, percentage, model_name, max_length, num_labels, seed=42):
    """
    Train the model with a specific percentage of the training data.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_samples = int(len(train_df) * percentage / 100)
    # Ensure n_samples is at least 2 and even (for binary balance)
    n_samples = max(2, n_samples - n_samples % 2)
    # Use balanced sampling instead of random sampling
    # We convert the Polars Data Frame to an arrow Dataset and get a sample of the training data
    if 'labels' in train_df.columns:
        train_df = train_df.rename({'labels': 'label'})  # Rename 'labels' to 'label' for compatibility with sampling function
    train_subset = sample_balanced_dataset(train_df, n_samples, seed)
    if 'label' in train_subset.column_names:
        train_subset = train_subset.rename_column('label', 'labels')  # Rename the label column again to 'labels' for compatibility with Hugging Face Trainer
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
        num_train_epochs=20,
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

def run_incremental_training(train_df, valid_df, model_name, max_length, num_labels, seed=42):
    """
    Run training with different percentages of the dataset.
    """
    percentages = [1, 10, 25, 50, 75, 100]
    all_results = []
    for pct in percentages:
        print(f"\nTraining with {pct}% of the data...")
        result = train_with_percentage(train_df, valid_df, pct, model_name, max_length, num_labels, seed)
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
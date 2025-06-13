from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
from transformers import Trainer
import polars as pl 
import numpy as np 

# Load and evaluate the best fine-tuned model on test set
def evaluate_best_finetuned_model(
    test_df: pl.DataFrame, 
    model_checkpoint_path: str, 
    original_model_name: str,
    max_length: int,
    sample_size: int = None,
    seed: int = 42
):
    """
    Load the best fine-tuned model from a checkpoint and evaluate it on the test set.

    Args:
        test_df (pl.DataFrame): The Polars DataFrame containing the test data. 
                                Must include 'text' and 'labels' columns.
        model_checkpoint_path (str): Path to the fine-tuned model checkpoint directory.
        original_model_name (str): Name or path of the tokenizer used during training.
        max_length (int): Maximum sequence length for tokenization.
        sample_size (int, optional): Number of samples to use for evaluation. 
                                     If None, the entire test set will be used.
        seed (int, optional): Random seed for reproducibility when sampling the test set.

    Returns:
        tuple: A tuple containing:
            - y_pred_test (np.ndarray): Predicted labels for the test set.
            - finetuned_probas (np.ndarray): Probabilities for the positive class.
    """

    print(f"Loading model from: {model_checkpoint_path}")
    print(f"Loading tokenizer from: {original_model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    # Load model from the specified checkpoint
    best_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)
    
    if sample_size:
        # Sample the test set if sample_size is specified
        print(f"Sampling {sample_size} rows from the test set...")
        test_df = test_df.sample(n=sample_size, shuffle=True, seed=seed)

    # Prepare test data: select only necessary columns
    # Assumes test_df already contains 'text' and 'labels' columns
    test_data_selected = test_df.select(['text', 'labels'])
    
    y_true = test_df['labels'].to_list()

    # Create test dataset from Polars DataFrame
    test_set = Dataset.from_polars(test_data_selected)
    
    # Tokenize function for the test set
    def tokenize_test(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)
    
    # Tokenize test set
    # remove_columns=["text"] ensures the original text column is removed after tokenization,
    # as the model expects 'input_ids', 'attention_mask', etc.
    test_set = test_set.map(tokenize_test, batched=True, remove_columns=["text"])
    test_set.set_format(type='torch')
    
    # Create trainer for evaluation (no training arguments needed for prediction)
    trainer = Trainer(model=best_model)
    
    # Get predictions
    print("Making predictions on the test set...")
    predictions_output = trainer.predict(test_set)
    
    # Extract predictions and probabilities
    # predictions_output.predictions contains the logits
    logits = torch.tensor(predictions_output.predictions)
    y_pred_proba = torch.softmax(logits, dim=1)
    y_pred_test = torch.argmax(y_pred_proba, dim=1).numpy()
    
    # Extract probabilities for the positive class (assuming label 1 is positive)
    # Ensure that the positive class is indeed at index 1.
    if y_pred_proba.shape[1] > 1:
        finetuned_probas = y_pred_proba[:, 1].numpy()
    else: # Handle cases where there might be only one output probability (e.g. regression, or misconfigured binary)
        finetuned_probas = y_pred_proba[:, 0].numpy() # Or handle as an error/warning
        print("Warning: Model output has only one dimension for probabilities. Assuming this is the positive class probability.")

    return y_true, y_pred_test, finetuned_probas
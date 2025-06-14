import polars as pl
import os
import setfit
from datasets import Dataset
from ..utilities import sample_balanced_dataset
from ..data_augmentation import augment_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from datetime import datetime

# Function to run SetFit training and evaluation routine
def run_setfit_training(train_df: pl.DataFrame, val_df: pl.DataFrame, 
                        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        num_epochs=5, batch_size=16, learning_rate=2e-5, sample_size=32,
                        metric='f1', num_iterations=10, seed=42, sample_proportion=0.5,
                        augmentation_rate:float=None, augmentation_techniques:list=None,
                        output_path:str=os.path.join('part_2', 'a'), custom_model_name_suffix:str=None):
    """
    Run SetFit training and evaluation routine.
    
    Args:
        train_df (pl.DataFrame): Training DataFrame with 'text' and 'label' columns.
        val_df (pl.DataFrame): Validation DataFrame with 'text' and 'label' columns.
        model_name (str): Pretrained model name for SetFit.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        sample_size (int): Number of samples to use for training in each iteration.
        metric (str): Metric to optimize during training ('f1', 'accuracy', etc.).
        num_iterations (int): Number of iterations to run the training process.
        seed (int): Random seed for reproducibility.
        sample_proportion (float): Proportion of positive samples to include in the balanced dataset.
        augmentation_rate (float): Rate of augmentations to apply per each augmentation 
            technique to the training data,expressed as a proportion of the dataset size 
            (e.g., 0.2 for 20%). This determines how many augmented examples will be generated.
            For example, if augmentation_rate is 0.2, the sample_size is 100, and the
            number of augmentation techniques is 2, then the total number of labelled
            examples in the training set will be 100 + (0.2 * 100 * 2) = 140.
        augmentation_techniques (list): List of initialized augmentation objects 
            to apply to the training data (like RandomDeletion, RandomInsertion, etc.).
        output_path (str): Intermediate path to save the results and model. Defaults to 'part_2/a'.
        custom_model_name_suffix (str): Custom suffix for the model name to avoid overwriting.
    
    Returns:
        list: A list of dictionaries containing evaluation metrics for each iteration.
    """
    
    # Prepare the validation set
    val_set = Dataset.from_polars(val_df.select(['text', 'label']))

    # Store results across iterations (for different metrics and iterations)
    iteration_results = []
    best_score = 0
    best_model = None
    best_iteration = 0

    # Run the sampling and training process with SetFit
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        # Create a fresh model for each iteration to avoid contamination
        model = setfit.SetFitModel.from_pretrained(model_name)

        # Sample balanced training data
        train_samples = sample_balanced_dataset(train_df, sample_size, seed + iteration, sample_proportion)

        if augmentation_rate is not None and augmentation_techniques is not None:
            train_samples = augment_dataset(
                dataset=train_samples,
                techniques=augmentation_techniques,
                augmentation_rate=augmentation_rate,
                seed=seed + iteration  # Ensure reproducibility across iterations
            )

        # Create the training arguments
        train_args = setfit.TrainingArguments(
            num_epochs=(num_epochs, num_epochs),  # Tuple format: (sentence_transformer_epochs, head_epochs)
            batch_size=(batch_size, batch_size),  # Tuple format: (sentence_transformer_batch, head_batch)
            body_learning_rate=(learning_rate, learning_rate),  # Tuple format for body learning rate (first for sentence transformer, second for head classifier)
        )

        # Initialize and train SetFit model
        trainer = setfit.Trainer(
            model=model,
            train_dataset=train_samples,  # Pairs of text and labels for Contrastive Learning
            eval_dataset=val_set,  # Validation set for evaluation
            metric=metric,  # Metric to optimize
            args=train_args  # Training arguments
        )

        trainer.train()  # Train the model
        print("Training completed.")

        # Evaluate on validation set (for hyperparameter tuning/model selection)
        val_predictions = trainer.model.predict(val_set['text'])
        val_metrics = {
            'accuracy': accuracy_score(val_set['label'], val_predictions),
            'f1': f1_score(val_set['label'], val_predictions),
            'precision': precision_score(val_set['label'], val_predictions),
            'recall': recall_score(val_set['label'], val_predictions)
        }
        
        # Store results for this iteration
        iteration_results.append(val_metrics)
        print(f"Validation F1: {val_metrics['f1']:.4f}")

        # Check if this is the best model so far
        current_score = val_metrics[metric]
        if current_score > best_score:
            best_score = current_score
            best_iteration = iteration + 1
            # Clean up previous best model
            if best_model is not None:
                del best_model
            # Store reference to current best model
            best_model = trainer.model
            print(f"New best model found with {metric}: {best_score:.4f}")

        # Clean up memory for this iteration - SINGLE, CLEAR LOGIC
        if best_model is trainer.model:
            # This is the best model, only delete trainer
            del model, trainer
        else:
            # This is not the best model, delete everything
            del model, trainer

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
    
    print('Finished training in all iterations. Saving the results to a parquet file...')
    
    # Save the best model
    san_model_name = model_name.split(sep='/')[-1]  # Sanitize model name for file path, keep only the last part
    if custom_model_name_suffix:
        final_folder_name = f'setfit_best_{san_model_name}_{custom_model_name_suffix}'
    else:
        final_folder_name = f'setfit_best_{san_model_name}'
    model_path = os.path.join('models', output_path, final_folder_name)

    timestamp_overwrite = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp for overwriting existing models

    # Check if model path exists and, if so, change naming to avoid overwriting
    if os.path.exists(model_path):
        timestamp = timestamp_overwrite
        final_folder_name = f'{final_folder_name}_{timestamp}'
        model_path = os.path.join('models', output_path, final_folder_name)
        print(f'Model path already exists. Using timestamp suffix: {final_folder_name}')

    os.makedirs(model_path, exist_ok=True)
    best_model.save_pretrained(model_path)  # Save the best model to the specified path
    print(f'Best model saved to: {model_path}')

    # Save the results in a polars DataFrame
    results_df = pl.DataFrame(iteration_results)
    results_df = results_df.with_columns(pl.Series("iteration", range(1, len(results_df) + 1)))  # Add iteration number to the results DataFrame

    # Save the results DataFrame to a Parquet file
    if custom_model_name_suffix:
        parquet_name = f'setfit_results_{san_model_name}_{custom_model_name_suffix}.parquet'
    else:
        parquet_name = f'setfit_results_{san_model_name}.parquet'

    results_path = os.path.join('results', output_path, parquet_name)

    # Check if results file exists and, if so, change naming to avoid overwriting
    if os.path.exists(results_path):
        timestamp = timestamp_overwrite
        base_name = parquet_name.replace('.parquet', '')
        parquet_name = f'{base_name}_{timestamp}.parquet'
        results_path = os.path.join('results', output_path, parquet_name)
        print(f'Results file already exists. Using timestamp suffix: {parquet_name}')

    os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Ensure the directory exists
    results_df.write_parquet(results_path)
    print(f'Results saved to: {results_path}')

    return results_df
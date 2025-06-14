import setfit

def evaluate_setfit_on_test(model_path, test_df, sample_size, seed):
    """
    Evaluate a trained SetFit model on the test set.
    
    Args:
        model_path (str): Path to the model used for training
        test_df (pl.DataFrame): (Pre-processed) Test dataset in Polars format.
            Ensure that the test DataFrame has been pre-processed similarly to 
            the training data, and that it contains the 'text' and 'label' columns.
        sample_size (int): Number of samples to use for evaluation. If None,
            the entire test set will be used.
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: True labels, predicted labels and predicted 
            probabilities for the positive class.
    """
    
    try:
        # Load the saved model
        best_setfit_model = setfit.SetFitModel.from_pretrained(model_path)
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")

    if sample_size:
        # Create a sample of the test set
        test_df = test_df.sample(n=sample_size, shuffle=True, seed=seed)  # Sample up to 1000 rows for evaluation

    # Extract text and labels for prediction
    test_texts = test_df['text'].to_list()
    y_true = test_df['label'].to_list()
    
    # Make predictions on the test set
    print("Making predictions on test set...")
    test_predictions = best_setfit_model.predict(test_texts)
    y_pred_labels = test_predictions.tolist()  # Convert predictions to a list

    # Get prediction probabilities
    test_probabilities = best_setfit_model.predict_proba(test_texts)
    test_probabilities_list = test_probabilities.tolist()  # Convert probabilities to a list

    # Extract probabilities for the positive class
    y_pred_probas = test_probabilities[:, 1]  # Extract the positive label (index 1) probabilities
    
    return y_true, y_pred_labels, y_pred_probas
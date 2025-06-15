import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_curve, auc, precision_recall_curve, 
                            average_precision_score)

def evaluate_classification(y_true: list[int], y_pred: list[int], base_filename, labels: list[int]=None, probas=None, results_dir="results"):
    """
    Evaluates classification performance and saves visualization results
    
    Args:
        y_true : list or array
            Ground truth labels. Should be a list of integers representing the true labels.
        y_pred : list or array
            Predicted labels. Should be a list of integers representing the predicted labels.
        base_filename : str
            Base name for saved files (e.g. "zero_shot", "few_shot")
        labels : list, optional
            List of class labels. Should be a list of integers representing the class labels.
            E.g., 0 for negative class (dismissal) and 1 for positive (approval).
            If None, it will be inferred from y_true.
        probas : array-like, optional
            Predicted probabilities for the positive class for ROC and PR curves.
            Should be a list or array of floats representing the predicted probabilities.
        results_dir : str, optional
            Directory to save results
    Returns:
        dict: Dictionary containing metrics and report
    """
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Ensure labels are provided
    if labels is None:
        labels = sorted(list(set(y_true)))
    
    # Check if model predicts only one class
    unique_predicted = set(y_pred)
    if len(unique_predicted) == 1:
        print(f"Warning: Model only predicts one class: {unique_predicted}")

    # 1. Classification report with accuracy
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)  # Metric is set to zero for undefined cases
    print(report)
    
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {base_filename}')
    plt.tight_layout()
    cm_path = f"{results_dir}/{base_filename}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=400)
    plt.show()
    plt.close()
    
    results = {
        "report": report,
        "saved_files": [cm_path]
    }
    
    # Process probabilities for ROC and PR curves if provided
    if probas is not None:
        # Convert labels to binary format for the curves
        unique_labels = sorted(list(set(y_true)))
        print(f"Unique labels found: {unique_labels}")
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        print(f"Label to index mapping: {label_to_idx}")
        y_binary = np.array([label_to_idx[y] for y in y_true])
        print("First 5 TRUE sorbinary labels for ROC/PR curves:", y_binary[:5])
        y_score = probas
        print("First 5 predicted probabilities for ROC/PR curves:", y_score[:5])
        
        # 3. ROC curve
        fpr, tpr, _ = roc_curve(y_binary, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {base_filename}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = f"{results_dir}/{base_filename}_roc_curve.png"
        plt.savefig(roc_path, dpi=400)
        plt.show()
        plt.close()
        results["saved_files"].append(roc_path)
        
        # 4. Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_binary, y_score, pos_label=1)
        avg_precision = average_precision_score(y_binary, y_score)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {base_filename}')
        plt.legend(loc="lower left")
        plt.tight_layout()
        pr_path = f"{results_dir}/{base_filename}_pr_curve.png"
        plt.savefig(pr_path, dpi=400)
        plt.show()
        plt.close()
        results["saved_files"].append(pr_path)
        
        # Add AUC metrics to results
        results["roc_auc"] = roc_auc
        results["avg_precision"] = avg_precision
    
    return results
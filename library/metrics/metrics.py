from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Metrics:
    def __init__(self):
        """Initializes the Metrics class with an empty dictionary to store results."""
        self.results = {}

    def run(self, y_true, y_pred, method_name, average='binary'):
        """
        Computes and stores evaluation metrics for a given set of predictions.

        Args:
            y_true: Array-like of true target values.
            y_pred: Array-like of predicted target values.
            method_name (str): Name of the method/model being evaluated.
            average (str): Averaging method for multi-class metrics ('binary', 'micro', 'weighted').
                           Defaults to 'binary'.
        """
        accuracy = accuracy_score(y_true, y_pred) * 100
        # Use zero_division=0 to avoid warnings when a class has no predictions
        precision = precision_score(y_true, y_pred, average=average, zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average=average, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0) * 100

        self.results[method_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        print(f"Metrics calculated for: {method_name}")

    def print_results(self):
        """
        Prints the stored metrics in a formatted table.
        """
        if not self.results:
            print("No metrics data to display. Use the .run() method first.")
            return

        for method, metrics in self.results.items():
            print("\n"+"="*40)
            print(f"Metrics for {method}")
            print("="*40)
            for metric, value in metrics.items():
                # Skip confusion matrices and other non-scalar values
                if metric == 'confusion_matrix' or hasattr(value, 'shape') and len(value.shape) > 0:
                    continue
                
                print(f"\n{metric}: {value:.2f}%")
                    
    def plot_confusion_matrix(self, y_true, y_pred, method_name, ax=None, class_labels=None, cmap='viridis'):
        """
        Computes and plots a confusion matrix for a given set of predictions.

        Args:
            y_true: Array-like of true target values.
            y_pred: Array-like of predicted target values.
            method_name (str): Name of the method/model for labeling.
            ax (matplotlib.axes, optional): Matplotlib axes to plot on. If None, a new figure is created.
            class_labels (list, optional): Labels for the classes in the confusion matrix.
                                        Defaults to ['Class 0', 'Class 1'] for binary classification.
            cmap (str): Colormap for the heatmap. Defaults to 'Blues'.
            
        Returns:
            matplotlib.axes: The axes with the plotted confusion matrix.
            numpy.ndarray: The confusion matrix.
        """
        
        # Compute confusion matrix
        conf_mat = confusion_matrix(y_true, y_pred)
        
        # Set default class labels for binary classification if not provided
        if class_labels is None:
            if set(np.unique(y_true)) == {0, 1}:
                class_labels = ['Dismissal (0)', 'Approval (1)']
            else:
                class_labels = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        
        # Create figure if no axes is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap=cmap,
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    ax=ax)
        
        # Set labels and title
        ax.set_title(f'Confusion Matrix - {method_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        return ax, conf_mat

    def plot_all_confusion_matrices(self, splits=None, figsize=(18, 6)):
        """
        Plots confusion matrices for all methods or specific splits.
        
        Args:
            splits (list, optional): List of split names to filter (e.g., ['train', 'validation', 'test']).
                                    If None, plots all stored confusion matrices.
            figsize (tuple): Figure size as (width, height).
            
        Returns:
            matplotlib.figure.Figure: The generated figure with confusion matrices.
        """
        # Filter methods based on splits if provided
        methods_to_plot = []
        if splits:
            for method in self.results.keys():
                for split in splits:
                    if split in method and 'confusion_matrix' in self.results[method]:
                        methods_to_plot.append(method)
        else:
            methods_to_plot = [m for m in self.results.keys() if 'confusion_matrix' in self.results[m]]
        
        if not methods_to_plot:
            print("No confusion matrices to plot.")
            return None
        
        # Create figure with subplots
        n_plots = len(methods_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        # Handle the case of a single subplot
        if n_plots == 1:
            axes = [axes]
        
        # Plot each confusion matrix
        for i, method in enumerate(methods_to_plot):
            conf_mat = self.results[method]['confusion_matrix']
            
            # For binary classification, use default labels
            if conf_mat.shape == (2, 2):
                class_labels = ['Dismissal (0)', 'Approval (1)']
            else:
                class_labels = [f'Class {j}' for j in range(conf_mat.shape[0])]
            
            ax = axes[i]
            sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_labels,
                        yticklabels=class_labels,
                        ax=ax)
            
            ax.set_title(f'{method}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        plt.tight_layout()
        return fig

    def plot(self):
        """
        Generates and displays a 2x2 grid of bar plots comparing the stored metrics
        across different methods.
        """
        if not self.results:
            print("No metrics data to plot. Use the .run() method first.")
            return

        methods = list(self.results.keys())
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()  # Flatten the 2x2 grid into a 1D array

        for i, metric_name in enumerate(metric_names):
            scores = [self.results[method][metric_name] for method in methods]

            bars = axes[i].bar(methods, scores, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
            axes[i].set_title(metric_name)
            axes[i].set_ylabel("Score (%)")
            axes[i].set_ylim(0, 105) # Set ylim to give space for annotations
            axes[i].tick_params(axis='x', rotation=45) # Rotate x-labels if they overlap

            # Add value annotations above each bar
            for bar in bars:
                yval = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}',
                                va='bottom', ha='center') # Adjust position slightly above bar

        plt.suptitle("Model Performance Comparison", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()
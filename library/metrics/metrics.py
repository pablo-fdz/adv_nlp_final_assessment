from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

class Metrics:
    def __init__(self):
        """Initializes the Metrics class with an empty dictionary to store results."""
        self.results = {}

    def run(self, y_true, y_pred, method_name, average='macro'):
        """
        Computes and stores evaluation metrics for a given set of predictions.

        Args:
            y_true: Array-like of true target values.
            y_pred: Array-like of predicted target values.
            method_name (str): Name of the method/model being evaluated.
            average (str): Averaging method for multi-class metrics ('macro', 'micro', 'weighted').
                           Defaults to 'macro'.
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
                print(f"\n{metric}: {value:.2f}%")

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

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten the 2x2 grid into a 1D array

        for i, metric_name in enumerate(metric_names):
            scores = [self.results[method][metric_name] for method in methods]

            bars = axes[i].bar(methods, scores, color=plt.cm.paired(np.linspace(0, 1, len(methods))))  # Check out color map options here: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps
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